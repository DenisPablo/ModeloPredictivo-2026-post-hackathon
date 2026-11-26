"""Export the trained 24h fire-risk model for FastAPI backend consumption.

Produces two backend-ready artifacts in `models/`:
1. `models/fire_risk_bundle.joblib`: Self-contained inference bundle (LightGBM
   booster string, feature order, Bayesian prior `logit_offset`, and percentile
   `risk_cutoffs`) loadable in FastAPI via `FireRiskPredictor.from_bundle(...)`.
2. `models/riesgo_celda_7d.parquet`: Precomputed ZSTD Parquet table for all
   64,653 active 5x5 km cells in Argentina (following `BACKEND_CONTRACTS.md`
   for `GET /api/v1/forecast` in FHIAPI + DuckDB).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import joblib
import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from modelo_predictivo.train_predictive_risk import (
    FEATURE_COLUMNS,
    categorize_risk,
    to_real_probability,
)


@dataclass
class FireRiskPredictor:
    """Self-contained predictor for FastAPI runtime or batch workers."""

    model_version: str
    feature_columns: list[str]
    logit_offset: float
    risk_cutoffs: dict[str, float]
    model_str: str

    def __post_init__(self) -> None:
        self._booster = lgb.Booster(model_str=self.model_str)

    @classmethod
    def from_model_dir(cls, model_dir: Path = Path("models")) -> FireRiskPredictor:
        """Create predictor from `fire_risk_lgbm.txt` and `fire_risk_metadata.json`."""
        model_path = model_dir / "fire_risk_lgbm.txt"
        meta_path = model_dir / "fire_risk_metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        cats = meta["risk_categories"]
        cutoffs = {
            name: float(cats[name]["min_probability"])
            for name in ("Extremo", "Alto", "Moderado")
        }
        return cls(
            model_version=str(meta["model_version"]),
            feature_columns=list(meta["features"]),
            logit_offset=float(meta["probability"]["logit_offset"]),
            risk_cutoffs=cutoffs,
            model_str=model_path.read_text(encoding="utf-8"),
        )

    @classmethod
    def from_bundle(cls, bundle_path: Path = Path("models/fire_risk_bundle.joblib")) -> FireRiskPredictor:
        """Load self-contained predictor from a single `.joblib` artifact."""
        payload = joblib.load(bundle_path)
        return cls(
            model_version=payload["model_version"],
            feature_columns=payload["feature_columns"],
            logit_offset=payload["logit_offset"],
            risk_cutoffs=payload["risk_cutoffs"],
            model_str=payload["model_str"],
        )

    def save_bundle(self, bundle_path: Path = Path("models/fire_risk_bundle.joblib")) -> Path:
        """Persist self-contained `.joblib` bundle for FastAPI deployment."""
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_version": self.model_version,
            "feature_columns": self.feature_columns,
            "logit_offset": self.logit_offset,
            "risk_cutoffs": self.risk_cutoffs,
            "model_str": self.model_str,
        }
        joblib.dump(payload, bundle_path, compress=3)
        return bundle_path

    def predict_matrix(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return `(probabilidad_riesgo, riesgo_pct, categoria_riesgo)` arrays."""
        raw = self._booster.predict(np.asarray(X, dtype=np.float32), raw_score=True)
        prob = to_real_probability(raw, self.logit_offset).astype(np.float32)
        pct = np.round(prob * 100.0, 4).astype(np.float32)
        cats = categorize_risk(prob, self.risk_cutoffs)
        return prob, pct, cats

    def predict_one(self, features: dict[str, float]) -> dict[str, Any]:
        """Score a single cell dictionary and return FastAPI-ready fields."""
        row = np.array([[float(features[col]) for col in self.feature_columns]], dtype=np.float32)
        prob, pct, cats = self.predict_matrix(row)
        return {
            "modelo_version": self.model_version,
            "probabilidad_riesgo": round(float(prob[0]), 6),
            "riesgo_pct": round(float(pct[0]), 4),
            "categoria_riesgo": str(cats[0]),
        }


def export_precomputed_forecast_parquet(
    predictor: FireRiskPredictor,
    firms_parquet: Path,
    weather_glob: str,
    output_parquet: Path,
    horizon_days: int = 7,
) -> dict[str, Any]:
    """Generate precomputed `riesgo_celda_7d.parquet` for all active cells."""
    con = duckdb.connect(database=":memory:")
    con.execute("SET memory_limit = '2GB';")

    max_weather_date = con.sql(
        f"SELECT MAX(fecha) FROM read_parquet('{weather_glob}')"
    ).fetchall()[0][0]
    start_date_sql = f"DATE '{max_weather_date}' - INTERVAL {max(horizon_days - 1, 0)} DAY"
    end_date_sql = f"DATE '{max_weather_date}'"
    target_year = int(str(max_weather_date)[:4])

    query = f"""
    WITH pos_events AS (
        SELECT
            cell_id,
            fecha,
            n_harmonized::FLOAT AS n_harm,
            phi_harmonized::FLOAT AS phi_harm
        FROM read_parquet('{firms_parquet.as_posix()}')
        WHERE n_harmonized >= 1
          AND phi_harmonized >= 10.0
          AND fecha <= {end_date_sql}
    ),
    active_cells AS (
        SELECT
            cell_id::INT AS cell_id,
            ANY_VALUE(lat_centro)::FLOAT AS lat_centro,
            ANY_VALUE(lon_centro)::FLOAT AS lon_centro,
            (ROUND(ANY_VALUE(lat_centro) * 2.0) / 2.0)::FLOAT AS lat_clima,
            (ROUND(ANY_VALUE(lon_centro) / 0.625) * 0.625)::FLOAT AS lon_clima,
            FLOOR(ANY_VALUE(lat_centro) / 0.12)::INT AS lat_bin,
            FLOOR(ANY_VALUE(lon_centro) / 0.14)::INT AS lon_bin
        FROM read_parquet('{firms_parquet.as_posix()}')
        WHERE lat_centro IS NOT NULL AND lon_centro IS NOT NULL
        GROUP BY cell_id
    ),
    cell_neighbors AS (
        SELECT a.cell_id AS dst_id, b.cell_id AS src_id
        FROM active_cells a
        JOIN active_cells b
          ON b.lat_bin BETWEEN a.lat_bin - 1 AND a.lat_bin + 1
         AND b.lon_bin BETWEEN a.lon_bin - 1 AND a.lon_bin + 1
         AND a.cell_id != b.cell_id
        WHERE (
            6371.0 * 2.0 * ASIN(SQRT(
                POWER(SIN(RADIANS(b.lat_centro - a.lat_centro) / 2.0), 2)
                + COS(RADIANS(a.lat_centro)) * COS(RADIANS(b.lat_centro))
                  * POWER(SIN(RADIANS(b.lon_centro - a.lon_centro) / 2.0), 2)
            ))
        ) <= 12.0
    ),
    neighbor_fires_at_start AS (
        SELECT
            cn.dst_id AS cell_id,
            SUM(p.n_harm)::FLOAT AS focos_vecinos_1d
        FROM pos_events p
        JOIN cell_neighbors cn ON cn.src_id = p.cell_id
        WHERE p.fecha = {start_date_sql} - INTERVAL 1 DAY
        GROUP BY cn.dst_id
    ),
    cell_month_clim AS (
        SELECT
            cell_id,
            EXTRACT(MONTH FROM fecha)::SMALLINT AS mes,
            MEDIAN(n_harm)::FLOAT AS mediana_historica_mes
        FROM pos_events
        WHERE fecha < DATE '{target_year}-01-01'
        GROUP BY cell_id, EXTRACT(MONTH FROM fecha)
    ),
    weather_base AS (
        SELECT
            lat_clima::FLOAT AS lat_clima,
            lon_clima::FLOAT AS lon_clima,
            elevacion_m::FLOAT AS elevacion_m,
            fecha,
            COALESCE(temperature_2m_max, 22.0)::FLOAT AS temperature_2m_max,
            LEAST(GREATEST(COALESCE(relative_humidity_2m, 55.0), 0.0), 100.0)::FLOAT AS relative_humidity_2m,
            GREATEST(COALESCE(wind_speed_10m_max, 12.0), 0.0)::FLOAT AS wind_speed_10m_max,
            GREATEST(COALESCE(precipitation_sum, 0.0), 0.0)::FLOAT AS precipitation_sum,
            GREATEST(COALESCE(shortwave_radiation_sum, 15.0), 0.0)::FLOAT AS shortwave_radiation_sum,
            CASE WHEN COALESCE(precipitation_sum, 0.0) >= 1.0 THEN 1 ELSE 0 END AS is_rain_day
        FROM read_parquet('{weather_glob}')
        WHERE fecha BETWEEN {start_date_sql} - INTERVAL 60 DAY AND {end_date_sql}
    ),
    weather_vpd AS (
        SELECT
            *,
            SUM(is_rain_day) OVER (
                PARTITION BY lat_clima, lon_clima ORDER BY fecha
            ) AS rain_group,
            (0.6108 * EXP((17.27 * temperature_2m_max) / (temperature_2m_max + 237.3)))::FLOAT AS es_kpa,
            (
                0.6108 * EXP((17.27 * temperature_2m_max) / (temperature_2m_max + 237.3))
                * (1.0 - relative_humidity_2m / 100.0)
            )::FLOAT AS vpd_approx
        FROM weather_base
    ),
    weather_et0 AS (
        SELECT
            *,
            GREATEST(
                0.0,
                (
                    ((4098.0 * es_kpa / POWER(temperature_2m_max + 237.3, 2)) * (0.408 * 0.70 * shortwave_radiation_sum))
                    + (
                        0.067 * (900.0 / (temperature_2m_max + 273.0))
                        * (wind_speed_10m_max * (1000.0 / 3600.0) * 0.748)
                        * vpd_approx
                    )
                ) / (
                    (4098.0 * es_kpa / POWER(temperature_2m_max + 237.3, 2))
                    + 0.067 * (1.0 + 0.34 * (wind_speed_10m_max * (1000.0 / 3600.0) * 0.748))
                )
            )::FLOAT AS et0_approx
        FROM weather_vpd
    ),
    weather_features AS (
        SELECT
            lat_clima,
            lon_clima,
            elevacion_m,
            fecha,
            temperature_2m_max,
            relative_humidity_2m,
            wind_speed_10m_max,
            precipitation_sum,
            shortwave_radiation_sum,
            SUM(precipitation_sum) OVER (
                PARTITION BY lat_clima, lon_clima ORDER BY fecha
                ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
            )::FLOAT AS lluvia_acum_7d,
            SUM(precipitation_sum) OVER (
                PARTITION BY lat_clima, lon_clima ORDER BY fecha
                ROWS BETWEEN 14 PRECEDING AND CURRENT ROW
            )::FLOAT AS lluvia_acum_15d,
            SUM(precipitation_sum) OVER (
                PARTITION BY lat_clima, lon_clima ORDER BY fecha
                ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
            )::FLOAT AS lluvia_acum_30d,
            CASE
                WHEN is_rain_day = 1 THEN 0::FLOAT
                ELSE (ROW_NUMBER() OVER (
                    PARTITION BY lat_clima, lon_clima, rain_group ORDER BY fecha
                ) - 1)::FLOAT
            END AS dias_consecutivos_sin_lluvia,
            vpd_approx,
            et0_approx,
            SUM(precipitation_sum - et0_approx) OVER (
                PARTITION BY lat_clima, lon_clima ORDER BY fecha
                ROWS BETWEEN 14 PRECEDING AND CURRENT ROW
            )::FLOAT AS balance_hidrico_15d,
            SUM(precipitation_sum - et0_approx) OVER (
                PARTITION BY lat_clima, lon_clima ORDER BY fecha
                ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
            )::FLOAT AS balance_hidrico_30d,
            AVG(vpd_approx) OVER (
                PARTITION BY lat_clima, lon_clima ORDER BY fecha
                ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
            )::FLOAT AS vpd_mean_7d
        FROM weather_et0
    ),
    target_dates AS (
        SELECT DISTINCT fecha
        FROM weather_features
        WHERE fecha BETWEEN {start_date_sql} AND {end_date_sql}
    ),
    cell_memory_at_start AS (
        SELECT
            c.cell_id,
            COALESCE(SUM(CASE
                WHEN p.fecha = {start_date_sql} - INTERVAL 1 DAY
                THEN p.n_harm ELSE 0.0
            END), 0.0)::FLOAT AS focos_ultimos_1d,
            COALESCE(SUM(CASE
                WHEN p.fecha BETWEEN {start_date_sql} - INTERVAL 2 DAY AND {start_date_sql} - INTERVAL 1 DAY
                THEN p.n_harm ELSE 0.0
            END), 0.0)::FLOAT AS focos_ultimos_2d,
            COALESCE(SUM(CASE
                WHEN p.fecha BETWEEN {start_date_sql} - INTERVAL 7 DAY AND {start_date_sql} - INTERVAL 1 DAY
                THEN p.n_harm ELSE 0.0
            END), 0.0)::FLOAT AS focos_ultimos_7d,
            COALESCE(SUM(CASE
                WHEN p.fecha BETWEEN {start_date_sql} - INTERVAL 30 DAY AND {start_date_sql} - INTERVAL 1 DAY
                THEN p.n_harm ELSE 0.0
            END), 0.0)::FLOAT AS focos_ultimos_30d,
            COALESCE(SUM(CASE
                WHEN p.fecha BETWEEN {start_date_sql} - INTERVAL 7 DAY AND {start_date_sql} - INTERVAL 1 DAY
                THEN p.phi_harm ELSE 0.0
            END), 0.0)::FLOAT AS frp_ultimos_7d,
            COALESCE(SUM(CASE
                WHEN p.fecha BETWEEN {start_date_sql} - INTERVAL 30 DAY AND {start_date_sql} - INTERVAL 1 DAY
                THEN p.phi_harm ELSE 0.0
            END), 0.0)::FLOAT AS frp_ultimos_30d,
            LEAST(31.0, COALESCE(
                DATEDIFF('day', MAX(CASE WHEN p.fecha < {start_date_sql} THEN p.fecha END), {start_date_sql}),
                31.0
            ))::FLOAT AS dias_desde_foco_reciente,
            COALESCE(
                DATEDIFF('day', MAX(CASE WHEN p.fecha < {start_date_sql} THEN p.fecha END), {start_date_sql}) / 365.25,
                10.0
            )::FLOAT AS anios_desde_ultimo_fuego
        FROM active_cells c
        LEFT JOIN pos_events p
            ON c.cell_id = p.cell_id
           AND p.fecha < {start_date_sql}
        GROUP BY c.cell_id
    )
    SELECT
        ac.cell_id,
        ac.lat_centro,
        ac.lon_centro,
        ac.lat_clima,
        ac.lon_clima,
        td.fecha,
        EXTRACT(MONTH FROM td.fecha)::FLOAT AS mes,
        EXTRACT(DOY FROM td.fecha)::FLOAT AS dia_anio,
        cm.focos_ultimos_1d,
        cm.focos_ultimos_2d,
        cm.focos_ultimos_7d,
        cm.focos_ultimos_30d,
        cm.frp_ultimos_7d,
        cm.frp_ultimos_30d,
        cm.dias_desde_foco_reciente,
        (cm.focos_ultimos_1d / (cm.focos_ultimos_7d + 1.0))::FLOAT AS ratio_actividad_1d_7d,
        COALESCE(nf.focos_vecinos_1d, 0.0)::FLOAT AS focos_vecinos_1d,
        COALESCE(cmc.mediana_historica_mes, 0.0)::FLOAT AS mediana_historica_mes,
        cm.anios_desde_ultimo_fuego,
        wf.elevacion_m,
        wf.temperature_2m_max,
        wf.relative_humidity_2m,
        wf.wind_speed_10m_max,
        wf.precipitation_sum,
        wf.shortwave_radiation_sum,
        wf.lluvia_acum_7d,
        wf.lluvia_acum_15d,
        wf.lluvia_acum_30d,
        wf.dias_consecutivos_sin_lluvia,
        wf.vpd_approx,
        wf.et0_approx,
        wf.balance_hidrico_15d,
        wf.balance_hidrico_30d,
        wf.vpd_mean_7d
    FROM active_cells ac
    CROSS JOIN target_dates td
    JOIN cell_memory_at_start cm ON ac.cell_id = cm.cell_id
    JOIN weather_features wf
      ON ac.lat_clima = wf.lat_clima
     AND ac.lon_clima = wf.lon_clima
     AND td.fecha = wf.fecha
    LEFT JOIN cell_month_clim cmc
      ON ac.cell_id = cmc.cell_id
     AND EXTRACT(MONTH FROM td.fecha)::SMALLINT = cmc.mes
    LEFT JOIN neighbor_fires_at_start nf
      ON ac.cell_id = nf.cell_id
    ORDER BY ac.cell_id, td.fecha
    """
    df_arrow = con.sql(query).to_arrow_table()
    con.close()

    X = np.column_stack(
        [df_arrow[col].to_numpy().astype(np.float32) for col in predictor.feature_columns]
    )
    prob, pct, cats = predictor.predict_matrix(X)
    generated_at = datetime.now(timezone.utc).isoformat()
    n_rows = df_arrow.num_rows

    out_table = pa.table(
        {
            "cell_id": df_arrow["cell_id"],
            "lat_centro": df_arrow["lat_centro"],
            "lon_centro": df_arrow["lon_centro"],
            "lat_clima": df_arrow["lat_clima"],
            "lon_clima": df_arrow["lon_clima"],
            "fecha": df_arrow["fecha"],
            "probabilidad_riesgo": pa.array(prob, type=pa.float32()),
            "riesgo_pct": pa.array(pct, type=pa.float32()),
            "categoria_riesgo": pa.array(cats.tolist(), type=pa.string()),
            "modelo_version": pa.array([predictor.model_version] * n_rows, type=pa.string()),
            "generado_en": pa.array([generated_at] * n_rows, type=pa.string()),
            "focos_ultimos_7d": df_arrow["focos_ultimos_7d"],
            "focos_ultimos_30d": df_arrow["focos_ultimos_30d"],
            "frp_ultimos_7d_mw": df_arrow["frp_ultimos_7d"],
            "mediana_historica_mes": df_arrow["mediana_historica_mes"],
            "anios_desde_ultimo_fuego": df_arrow["anios_desde_ultimo_fuego"],
            "elevacion_m": df_arrow["elevacion_m"],
            "temp_max_c": df_arrow["temperature_2m_max"],
            "rh_min_pct": df_arrow["relative_humidity_2m"],
            "wind_max_kmh": df_arrow["wind_speed_10m_max"],
            "precipitation_sum_mm": df_arrow["precipitation_sum"],
            "shortwave_radiation_sum_mj_m2": df_arrow["shortwave_radiation_sum"],
            "dias_consecutivos_sin_lluvia": df_arrow["dias_consecutivos_sin_lluvia"],
            "vpd_approx_kpa": df_arrow["vpd_approx"],
        }
    )
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_table, output_parquet, compression="zstd")

    unique_cats, counts = np.unique(cats, return_counts=True)
    return {
        "output_parquet": str(output_parquet),
        "rows": int(n_rows),
        "size_mb": round(output_parquet.stat().st_size / 1e6, 2),
        "category_counts": {str(k): int(v) for k, v in zip(unique_cats, counts)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export fire-risk model bundle and precomputed Parquet for FastAPI."
    )
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--firms",
        type=Path,
        default=Path("data/firms_argentina_harmonized.parquet"),
    )
    parser.add_argument(
        "--weather-glob",
        type=str,
        default="data/weather/nasapower/nasapower_*.parquet",
    )
    parser.add_argument(
        "--bundle-out",
        type=Path,
        default=Path("models/fire_risk_bundle.joblib"),
    )
    parser.add_argument(
        "--parquet-out",
        type=Path,
        default=Path("models/riesgo_celda_7d.parquet"),
    )
    parser.add_argument("--horizon-days", type=int, default=7)
    args = parser.parse_args()

    predictor = FireRiskPredictor.from_model_dir(args.model_dir)
    bundle_path = predictor.save_bundle(args.bundle_out)
    parquet_info = export_precomputed_forecast_parquet(
        predictor=predictor,
        firms_parquet=args.firms,
        weather_glob=args.weather_glob,
        output_parquet=args.parquet_out,
        horizon_days=args.horizon_days,
    )
    print(
        json.dumps(
            {
                "bundle_path": str(bundle_path),
                "bundle_size_kb": round(bundle_path.stat().st_size / 1e3, 1),
                "model_version": predictor.model_version,
                "precomputed_forecast": parquet_info,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
