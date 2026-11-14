"""Train the 24h predictive fire risk model (LightGBM) for Argentina.

Combines harmonized FIRMS fire history (`firms_argentina_harmonized.parquet`)
with NASA POWER daily weather (`data/weather/nasapower/nasapower_*.parquet`)
using strict Time-Series Split (train <= 2023, val = 2024-2025) and leak-free
fire-memory and drought features.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import lightgbm as lgb
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

FEATURE_COLUMNS = [
    "lat_centro",
    "lon_centro",
    "mes",
    "dia_anio",
    "focos_ultimos_7d",
    "focos_ultimos_30d",
    "frp_ultimos_7d",
    "mediana_historica_mes",
    "anios_desde_ultimo_fuego",
    "elevacion_m",
    "temperature_2m_max",
    "relative_humidity_2m",
    "wind_speed_10m_max",
    "precipitation_sum",
    "shortwave_radiation_sum",
    "lluvia_acum_7d",
    "lluvia_acum_15d",
    "lluvia_acum_30d",
    "dias_consecutivos_sin_lluvia",
    "vpd_approx",
]


def format_duration(seconds: float) -> str:
    """Format seconds into compact HH:MM:SS string."""
    secs = max(0, int(seconds))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def render_progress(
    done: int,
    total: int,
    label: str,
    start_ts: float,
    extra: str = "",
) -> None:
    """Print an in-place terminal progress bar with percentage and ETA."""
    pct = (done / max(total, 1)) * 100.0
    bar_len = 28
    filled = int(bar_len * done / max(total, 1))
    bar = "#" * filled + "-" * (bar_len - filled)

    elapsed = time.perf_counter() - start_ts
    if done > 0:
        eta_str = format_duration((elapsed / done) * (total - done))
    else:
        eta_str = "--:--:--"

    suffix = f" | {extra}" if extra else ""
    line = (
        f"\r[{bar}] {done}/{total} ({pct:5.1f}%) | "
        f"{label}{suffix} | elapsed={format_duration(elapsed)} | ETA={eta_str}"
    )
    sys.stdout.write(line)
    sys.stdout.flush()


def build_dataset_splits(
    firms_parquet: Path,
    weather_glob: str,
    train_end: str = "2023-12-31",
    val_start: str = "2024-01-01",
    val_end: str = "2025-12-31",
    test_start: str | None = "2026-01-01",
    test_end: str | None = "2026-12-31",
    neg_sample_permille: int = 8,
    only_years: list[int] | None = None,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
]:
    """Build leak-free train/val/test feature matrices in float32 via DuckDB (batched by year)."""
    t0 = time.perf_counter()
    print("[1/3] Preparing DuckDB tables (FIRMS + NASA POWER weather features)...")
    threads = min(8, os.cpu_count() or 4)
    data_end = test_end if (test_start and test_end) else val_end

    with tempfile.TemporaryDirectory(prefix="duckdb_spill_") as spill_dir:
        con = duckdb.connect(database=":memory:")
        con.execute(f"SET threads TO {threads};")
        con.execute("SET memory_limit = '2GB';")
        con.execute(f"SET temp_directory = '{Path(spill_dir).as_posix()}';")

        con.execute(f"""
        CREATE TEMP TABLE pos_events AS
        SELECT
            cell_id,
            fecha,
            n_harmonized::FLOAT AS n_harm,
            phi_harmonized::FLOAT AS phi_harm
        FROM read_parquet('{firms_parquet.as_posix()}')
        WHERE n_harmonized >= 1
          AND phi_harmonized >= 10.0
          AND fecha BETWEEN DATE '2011-01-01' AND DATE '{data_end}';

        CREATE TEMP TABLE active_cells AS
        SELECT
            cell_id,
            ANY_VALUE(lat_centro)::FLOAT AS lat_centro,
            ANY_VALUE(lon_centro)::FLOAT AS lon_centro,
            (ROUND(ANY_VALUE(lat_centro) * 2.0) / 2.0)::FLOAT AS lat_clima,
            (ROUND(ANY_VALUE(lon_centro) / 0.625) * 0.625)::FLOAT AS lon_clima
        FROM read_parquet('{firms_parquet.as_posix()}')
        WHERE lat_centro IS NOT NULL AND lon_centro IS NOT NULL
        GROUP BY cell_id;

        CREATE TEMP TABLE weather_features AS
        WITH weather_base AS (
            SELECT
                lat_clima::FLOAT AS lat_clima,
                lon_clima::FLOAT AS lon_clima,
                elevacion_m::FLOAT AS elevacion_m,
                fecha,
                temperature_2m_max::FLOAT AS temperature_2m_max,
                relative_humidity_2m::FLOAT AS relative_humidity_2m,
                wind_speed_10m_max::FLOAT AS wind_speed_10m_max,
                precipitation_sum::FLOAT AS precipitation_sum,
                shortwave_radiation_sum::FLOAT AS shortwave_radiation_sum,
                CASE WHEN COALESCE(precipitation_sum, 0.0) >= 1.0 THEN 1 ELSE 0 END AS is_rain_day
            FROM read_parquet('{weather_glob}')
            WHERE fecha BETWEEN DATE '2018-01-01' AND DATE '{data_end}'
        ),
        weather_streaks AS (
            SELECT
                *,
                SUM(is_rain_day) OVER (
                    PARTITION BY lat_clima, lon_clima ORDER BY fecha
                ) AS rain_group
            FROM weather_base
        )
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
            SUM(COALESCE(precipitation_sum, 0.0)) OVER (
                PARTITION BY lat_clima, lon_clima ORDER BY fecha
                ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
            )::FLOAT AS lluvia_acum_7d,
            SUM(COALESCE(precipitation_sum, 0.0)) OVER (
                PARTITION BY lat_clima, lon_clima ORDER BY fecha
                ROWS BETWEEN 14 PRECEDING AND CURRENT ROW
            )::FLOAT AS lluvia_acum_15d,
            SUM(COALESCE(precipitation_sum, 0.0)) OVER (
                PARTITION BY lat_clima, lon_clima ORDER BY fecha
                ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
            )::FLOAT AS lluvia_acum_30d,
            CASE
                WHEN is_rain_day = 1 THEN 0::FLOAT
                ELSE (ROW_NUMBER() OVER (
                    PARTITION BY lat_clima, lon_clima, rain_group ORDER BY fecha
                ) - 1)::FLOAT
            END AS dias_consecutivos_sin_lluvia,
            (
                0.6108 * EXP((17.27 * temperature_2m_max) / (temperature_2m_max + 237.3))
                * (1.0 - LEAST(GREATEST(relative_humidity_2m, 0.0), 100.0) / 100.0)
            )::FLOAT AS vpd_approx
        FROM weather_streaks;
        """)

        years = [
            int(r[0])
            for r in con.sql(
                "SELECT DISTINCT EXTRACT(YEAR FROM fecha)::INT AS y "
                "FROM weather_features ORDER BY y"
            ).fetchall()
        ]
        if only_years is not None:
            allowed = set(only_years)
            years = [y for y in years if y in allowed]

        x_train_chunks: list[np.ndarray] = []
        y_train_chunks: list[np.ndarray] = []
        f_train_chunks: list[np.ndarray] = []
        x_val_chunks: list[np.ndarray] = []
        y_val_chunks: list[np.ndarray] = []
        f_val_chunks: list[np.ndarray] = []
        x_test_chunks: list[np.ndarray] = []
        y_test_chunks: list[np.ndarray] = []
        f_test_chunks: list[np.ndarray] = []

        total_rows = 0
        batch_start = time.perf_counter()
        print(f"[2/3] Extracting features in {len(years)} yearly batches...")

        for idx, year in enumerate(years, start=1):
            y_start = f"{year}-01-01"
            y_end = min(f"{year}-12-31", data_end)
            batch_query = f"""
            WITH cell_month_clim AS (
                -- Climatology strictly from years BEFORE the batch year (no self-label leakage)
                SELECT
                    cell_id,
                    EXTRACT(MONTH FROM fecha)::SMALLINT AS mes,
                    MEDIAN(n_harm)::FLOAT AS mediana_historica_mes
                FROM pos_events
                WHERE fecha < DATE '{y_start}'
                GROUP BY cell_id, EXTRACT(MONTH FROM fecha)
            ),
            calendar_days AS (
                SELECT DISTINCT fecha
                FROM weather_features
                WHERE fecha BETWEEN DATE '{y_start}' AND DATE '{y_end}'
            ),
            neg_samples AS (
                SELECT
                    c.cell_id,
                    d.fecha,
                    0::TINYINT AS target
                FROM active_cells c
                CROSS JOIN calendar_days d
                LEFT JOIN pos_events p
                    ON c.cell_id = p.cell_id AND d.fecha = p.fecha
                WHERE p.cell_id IS NULL
                  AND (ABS(HASH(c.cell_id, d.fecha)) % 1000) < {neg_sample_permille}
            ),
            labeled_samples AS (
                SELECT cell_id, fecha, 1::TINYINT AS target
                FROM pos_events
                WHERE fecha BETWEEN DATE '{y_start}' AND DATE '{y_end}'
                UNION ALL
                SELECT cell_id, fecha, target
                FROM neg_samples
            ),
            fire_memory AS (
                -- Compute fire memory strictly prior to target fecha (<= t = fecha - 1 day)
                SELECT
                    s.cell_id,
                    s.fecha,
                    s.target,
                    COALESCE(SUM(CASE
                        WHEN p.fecha BETWEEN s.fecha - INTERVAL 7 DAY AND s.fecha - INTERVAL 1 DAY
                        THEN p.n_harm ELSE 0.0
                    END), 0.0)::FLOAT AS focos_ultimos_7d,
                    COALESCE(SUM(CASE
                        WHEN p.fecha BETWEEN s.fecha - INTERVAL 30 DAY AND s.fecha - INTERVAL 1 DAY
                        THEN p.n_harm ELSE 0.0
                    END), 0.0)::FLOAT AS focos_ultimos_30d,
                    COALESCE(SUM(CASE
                        WHEN p.fecha BETWEEN s.fecha - INTERVAL 7 DAY AND s.fecha - INTERVAL 1 DAY
                        THEN p.phi_harm ELSE 0.0
                    END), 0.0)::FLOAT AS frp_ultimos_7d,
                    COALESCE(
                        DATEDIFF('day', MAX(CASE WHEN p.fecha < s.fecha THEN p.fecha END), s.fecha) / 365.25,
                        10.0
                    )::FLOAT AS anios_desde_ultimo_fuego
                FROM labeled_samples s
                LEFT JOIN pos_events p
                    ON s.cell_id = p.cell_id
                   AND p.fecha < s.fecha
                GROUP BY s.cell_id, s.fecha, s.target
            )
            SELECT
                fm.fecha,
                fm.target,
                ac.lat_centro,
                ac.lon_centro,
                EXTRACT(MONTH FROM fm.fecha)::FLOAT AS mes,
                EXTRACT(DOY FROM fm.fecha)::FLOAT AS dia_anio,
                fm.focos_ultimos_7d,
                fm.focos_ultimos_30d,
                fm.frp_ultimos_7d,
                COALESCE(cmc.mediana_historica_mes, 0.0)::FLOAT AS mediana_historica_mes,
                fm.anios_desde_ultimo_fuego,
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
                wf.vpd_approx
            FROM fire_memory fm
            JOIN active_cells ac ON fm.cell_id = ac.cell_id
            JOIN weather_features wf
              ON ac.lat_clima = wf.lat_clima
             AND ac.lon_clima = wf.lon_clima
             AND fm.fecha = wf.fecha
            LEFT JOIN cell_month_clim cmc
              ON fm.cell_id = cmc.cell_id
             AND EXTRACT(MONTH FROM fm.fecha)::SMALLINT = cmc.mes
            """
            df_arrow = con.sql(batch_query).to_arrow_table()
            if df_arrow.num_rows > 0:
                fechas_b = df_arrow["fecha"].to_numpy()
                y_b = df_arrow["target"].to_numpy().astype(np.int32)
                x_b = np.column_stack(
                    [df_arrow[col].to_numpy().astype(np.float32) for col in FEATURE_COLUMNS]
                )

                tr_mask = fechas_b <= np.datetime64(train_end)
                va_mask = (fechas_b >= np.datetime64(val_start)) & (fechas_b <= np.datetime64(val_end))
                if tr_mask.any():
                    x_train_chunks.append(x_b[tr_mask])
                    y_train_chunks.append(y_b[tr_mask])
                    f_train_chunks.append(fechas_b[tr_mask])
                if va_mask.any():
                    x_val_chunks.append(x_b[va_mask])
                    y_val_chunks.append(y_b[va_mask])
                    f_val_chunks.append(fechas_b[va_mask])
                if test_start and test_end:
                    te_mask = (fechas_b >= np.datetime64(test_start)) & (fechas_b <= np.datetime64(test_end))
                    if te_mask.any():
                        x_test_chunks.append(x_b[te_mask])
                        y_test_chunks.append(y_b[te_mask])
                        f_test_chunks.append(fechas_b[te_mask])

                total_rows += df_arrow.num_rows

            render_progress(idx, len(years), f"year={year}", batch_start, f"rows={total_rows:,}")

        con.close()

    sys.stdout.write("\n")
    sys.stdout.flush()

    empty_x = np.empty((0, len(FEATURE_COLUMNS)), dtype=np.float32)
    empty_y = np.empty((0,), dtype=np.int32)
    empty_f = np.empty((0,), dtype="datetime64[D]")

    X_train = np.vstack(x_train_chunks) if x_train_chunks else empty_x
    y_train = np.concatenate(y_train_chunks) if y_train_chunks else empty_y
    f_train = np.concatenate(f_train_chunks) if f_train_chunks else empty_f
    X_val = np.vstack(x_val_chunks) if x_val_chunks else empty_x
    y_val = np.concatenate(y_val_chunks) if y_val_chunks else empty_y
    f_val = np.concatenate(f_val_chunks) if f_val_chunks else empty_f
    X_test = np.vstack(x_test_chunks) if x_test_chunks else empty_x
    y_test = np.concatenate(y_test_chunks) if y_test_chunks else empty_y
    f_test = np.concatenate(f_test_chunks) if f_test_chunks else empty_f

    print(
        f"Dataset ready in {format_duration(time.perf_counter() - t0)}: "
        f"train={len(y_train):,} | val={len(y_val):,} | test={len(y_test):,}"
    )
    return X_train, y_train, f_train, X_val, y_val, f_val, X_test, y_test, f_test


def build_training_dataset(
    firms_parquet: Path,
    weather_glob: str,
    train_end: str = "2023-12-31",
    val_start: str = "2024-01-01",
    val_end: str = "2025-12-31",
    neg_sample_permille: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build leak-free train/val feature matrices in float32 via DuckDB."""
    X_tr, y_tr, f_tr, X_va, y_va, f_va, _, _, _ = build_dataset_splits(
        firms_parquet=firms_parquet,
        weather_glob=weather_glob,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=None,
        test_end=None,
        neg_sample_permille=neg_sample_permille,
    )
    return X_tr, y_tr, f_tr, X_va, y_va, f_va


# Risk categories are defined by real-world percentile of cell-days (derived on
# validation), not by fixed probability cutoffs: real daily fire probability is
# very low (~0.1%), so fixed 0.25/0.50/0.75 cutoffs would be meaningless.
RISK_CATEGORY_TOP_FRACTIONS = {"Extremo": 0.001, "Alto": 0.01, "Moderado": 0.05}
TOP_K_FRACTIONS = (0.001, 0.01, 0.05)


def prior_logit_offset(neg_sample_permille: int, scale_pos_weight: float) -> float:
    """Logit shift that undoes negative downsampling and positive class weighting."""
    return float(np.log(neg_sample_permille / 1000.0) - np.log(scale_pos_weight))


def to_real_probability(raw_margin: np.ndarray, logit_offset: float) -> np.ndarray:
    """Convert LightGBM raw margins into real-world daily fire probabilities."""
    z = np.asarray(raw_margin, dtype=np.float64) + logit_offset
    return 1.0 / (1.0 + np.exp(-z))


def predict_real_probability(booster: lgb.Booster, X: np.ndarray, logit_offset: float) -> np.ndarray:
    """Predict real-world daily fire probabilities with a saved booster."""
    return to_real_probability(booster.predict(X, raw_score=True), logit_offset)


def population_weights(y_true: np.ndarray, neg_sample_permille: int) -> np.ndarray:
    """Weights that re-expand sampled negatives to the real cell-day population."""
    return np.where(y_true == 1, 1.0, 1000.0 / neg_sample_permille)


def weighted_top_cutoff(prob: np.ndarray, weights: np.ndarray, top_fraction: float) -> float:
    """Probability cutoff selecting the top `top_fraction` of the weighted population."""
    order = np.argsort(-prob)
    cum_w = np.cumsum(weights[order])
    k = min(int(np.searchsorted(cum_w, top_fraction * cum_w[-1])), len(order) - 1)
    return float(prob[order][k])


def derive_risk_cutoffs(prob: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    """Derive minimum probability per risk category from population percentiles."""
    return {
        name: weighted_top_cutoff(prob, weights, frac)
        for name, frac in RISK_CATEGORY_TOP_FRACTIONS.items()
    }


def categorize_risk(prob: np.ndarray, cutoffs: dict[str, float]) -> np.ndarray:
    """Assign Bajo/Moderado/Alto/Extremo labels using derived cutoffs."""
    labels = np.full(len(prob), "Bajo", dtype=object)
    for name in ("Moderado", "Alto", "Extremo"):
        labels[prob >= cutoffs[name]] = name
    return labels


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sample_weight: np.ndarray | None = None,
    risk_cutoffs: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compute real-world (population-weighted) ranking, calibration and lift metrics."""
    w = np.ones(len(y_true)) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    pos = (y_true == 1).astype(np.float64)
    total_w = float(w.sum())
    total_pos = float((pos * w).sum())
    base_rate = total_pos / total_w
    mean_pred = float(np.sum(y_prob * w) / total_w)

    metrics: dict[str, Any] = {
        "n_samples": int(len(y_true)),
        "positive_rate": base_rate,
        "mean_predicted_probability": mean_pred,
        "calibration_ratio": mean_pred / max(base_rate, 1e-12),
        "roc_auc": float(roc_auc_score(y_true, y_prob, sample_weight=w)),
        "pr_auc": float(average_precision_score(y_true, y_prob, sample_weight=w)),
        "brier_score": float(brier_score_loss(y_true, y_prob, sample_weight=w)),
        "log_loss": float(log_loss(y_true, np.clip(y_prob, 1e-12, 1 - 1e-12), sample_weight=w)),
        "top_k": {},
    }

    order = np.argsort(-y_prob)
    cum_w = np.cumsum(w[order])
    cum_pos = np.cumsum((pos * w)[order])
    for frac in TOP_K_FRACTIONS:
        k = min(int(np.searchsorted(cum_w, frac * total_w)), len(order) - 1)
        precision = float(cum_pos[k] / cum_w[k])
        metrics["top_k"][f"top_{frac * 100:g}pct"] = {
            "precision": precision,
            "recall": float(cum_pos[k] / max(total_pos, 1e-12)),
            "lift": precision / max(base_rate, 1e-12),
        }

    if risk_cutoffs is not None:
        labels = categorize_risk(y_prob, risk_cutoffs)
        metrics["categories"] = {}
        for name in ("Extremo", "Alto", "Moderado", "Bajo"):
            sel = labels == name
            cat_w = float(w[sel].sum())
            cat_pos = float((pos * w)[sel].sum())
            precision = cat_pos / cat_w if cat_w > 0 else 0.0
            metrics["categories"][name] = {
                "share_of_cell_days": cat_w / total_w,
                "precision": precision,
                "recall": cat_pos / max(total_pos, 1e-12),
                "lift": precision / max(base_rate, 1e-12),
                "mean_predicted_probability": float(np.sum(y_prob[sel] * w[sel]) / cat_w) if cat_w > 0 else 0.0,
            }
    return metrics


def recency_baseline_pr_auc(X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> float:
    """PR-AUC of the strongest single-feature baseline (recency of last fire)."""
    score = -X[:, FEATURE_COLUMNS.index("anios_desde_ultimo_fuego")]
    return float(average_precision_score(y, score, sample_weight=sample_weight))


def train_and_export(
    firms_parquet: Path,
    weather_glob: str,
    model_dir: Path,
    neg_sample_permille: int,
    test_start: str | None = "2026-01-01",
    test_end: str | None = "2026-12-31",
    scale_pos_weight: float = 1.0,
) -> dict[str, Any]:
    """Train LightGBM classifier and save model artifact + metadata."""
    X_train, y_train, _, X_val, y_val, _, X_test, y_test, _ = build_dataset_splits(
        firms_parquet=firms_parquet,
        weather_glob=weather_glob,
        test_start=test_start,
        test_end=test_end,
        neg_sample_permille=neg_sample_permille,
    )
    if len(y_train) == 0 or len(y_val) == 0:
        raise RuntimeError(
            f"Empty split detected: train={len(y_train)}, val={len(y_val)}."
        )

    pos_count = int(np.sum(y_train == 1))
    neg_count = int(np.sum(y_train == 0))
    raw_neg_pos_ratio = float(neg_count / max(pos_count, 1))
    scale_pos_weight = float(scale_pos_weight)

    print("[3/3] Training LightGBM model...")
    fit_start = time.perf_counter()
    n_estimators = 500

    def _progress_callback(env: lgb.callback.CallbackEnv) -> None:
        it = env.iteration + 1
        loss_str = ""
        if env.evaluation_result_list:
            loss_val = env.evaluation_result_list[0][2]
            loss_str = f"val_logloss={loss_val:.4f}"
        render_progress(it, n_estimators, f"iter={it}", fit_start, loss_str)

    clf = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=63,
        max_bin=255,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[
            lgb.early_stopping(stopping_rounds=40, first_metric_only=True, verbose=False),
            _progress_callback,
        ],
    )
    sys.stdout.write("\n")
    sys.stdout.flush()

    # Real-world probability: undo negative downsampling and class weighting.
    logit_offset = prior_logit_offset(neg_sample_permille, scale_pos_weight)
    w_val = population_weights(y_val, neg_sample_permille)
    p_val = to_real_probability(clf.predict(X_val, raw_score=True), logit_offset)
    # Category cutoffs are derived on validation only (2026 test stays untouched).
    risk_cutoffs = derive_risk_cutoffs(p_val, w_val)

    val_metrics = evaluate_predictions(y_val, p_val, w_val, risk_cutoffs)
    val_metrics["baseline_recency_pr_auc"] = recency_baseline_pr_auc(X_val, y_val, w_val)
    test_metrics = None
    if len(y_test) > 0:
        w_test = population_weights(y_test, neg_sample_permille)
        p_test = to_real_probability(clf.predict(X_test, raw_score=True), logit_offset)
        test_metrics = evaluate_predictions(y_test, p_test, w_test, risk_cutoffs)
        test_metrics["baseline_recency_pr_auc"] = recency_baseline_pr_auc(X_test, y_test, w_test)

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "fire_risk_lgbm.txt"
    meta_path = model_dir / "fire_risk_metadata.json"

    clf.booster_.save_model(str(model_path))

    importances = {
        col: int(imp)
        for col, imp in zip(FEATURE_COLUMNS, clf.feature_importances_)
    }
    metadata = {
        "model_version": "lgbm-risk-24h-v2",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "horizon_hours": 24,
        "positive_label_rule": "n_harmonized >= 1 AND phi_harmonized >= 10.0",
        "split": {
            "train": "2018-01-01..2023-12-31",
            "val": "2024-01-01..2025-12-31",
            "test": f"{test_start}..{test_end}" if test_metrics else None,
            "train_rows": int(len(y_train)),
            "val_rows": int(len(y_val)),
            "test_rows": int(len(y_test)),
            "train_pos": pos_count,
            "train_neg": neg_count,
            "raw_neg_pos_ratio": raw_neg_pos_ratio,
            "scale_pos_weight": float(scale_pos_weight),
            "best_iteration": int(clf.best_iteration_ or clf.n_estimators),
        },
        "probability": {
            "meaning": "Real-world probability of a positive fire label for the cell in the next 24h",
            "method": "prior_correction",
            "neg_sample_permille": int(neg_sample_permille),
            "logit_offset": logit_offset,
            "formula": "p = 1 / (1 + exp(-(booster.predict(X, raw_score=True) + logit_offset)))",
        },
        "features": FEATURE_COLUMNS,
        "feature_importance_split": importances,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "risk_categories": {
            "derived_on": "validation",
            "Extremo": {"top_fraction_of_cell_days": RISK_CATEGORY_TOP_FRACTIONS["Extremo"],
                        "min_probability": risk_cutoffs["Extremo"]},
            "Alto": {"top_fraction_of_cell_days": RISK_CATEGORY_TOP_FRACTIONS["Alto"],
                     "min_probability": risk_cutoffs["Alto"]},
            "Moderado": {"top_fraction_of_cell_days": RISK_CATEGORY_TOP_FRACTIONS["Moderado"],
                         "min_probability": risk_cutoffs["Moderado"]},
            "Bajo": {"top_fraction_of_cell_days": 1.0, "min_probability": 0.0},
        },
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train 24h fire-risk LightGBM model for Argentina."
    )
    parser.add_argument(
        "--firms",
        type=Path,
        default=Path("firms_argentina_harmonized.parquet"),
    )
    parser.add_argument(
        "--weather-glob",
        type=str,
        default="data/weather/nasapower/nasapower_*.parquet",
    )
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--neg-sample-permille", type=int, default=8)
    parser.add_argument(
        "--scale-pos-weight",
        type=float,
        default=1.0,
        help="Positive class weight (default 1.0: negative sampling already balances classes).",
    )
    args = parser.parse_args()

    metadata = train_and_export(
        firms_parquet=args.firms,
        weather_glob=args.weather_glob,
        model_dir=args.model_dir,
        neg_sample_permille=args.neg_sample_permille,
        scale_pos_weight=args.scale_pos_weight,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
