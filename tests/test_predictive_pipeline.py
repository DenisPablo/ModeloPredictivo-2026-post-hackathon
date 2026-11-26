"""Verification tests for the 24h predictive fire-risk pipeline (S1-S5)."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from modelo_predictivo.ingest_nasapower import ARGENTINA_TILES, NASA_SCHEMA
from modelo_predictivo.train_predictive_risk import (
    FEATURE_COLUMNS,
    build_training_dataset,
    train_and_export,
)


def _create_synthetic_firms(path: Path) -> None:
    rows = []
    # Cell 1: positive fire in train (2023-06-10), low-FRP (<10 MW) in 2024-06-15,
    # and high-FRP positive fire in val (2024-06-20).
    # Cell 2: positive fire in 2023-06-10 and 2024-06-19 (day before 2024-06-20).
    events = [
        (1, -34.89, -66.54, date(2023, 6, 10), 2, 25.0),
        (1, -34.89, -66.54, date(2023, 6, 11), 3, 40.0),
        (1, -34.89, -66.54, date(2024, 6, 15), 1, 5.0),   # filtered out (< 10 MW)
        (1, -34.89, -66.54, date(2024, 6, 20), 2, 30.0),  # positive in val
        (2, -33.10, -65.20, date(2023, 6, 10), 1, 15.0),
        (2, -33.10, -65.20, date(2024, 6, 19), 2, 20.0),  # day before 2024-06-20
        (2, -33.10, -65.20, date(2024, 6, 20), 2, 25.0),  # positive in val
    ]
    for cid, lat, lon, d, n_h, phi_h in events:
        rows.append(
            {
                "cell_id": cid,
                "lat_centro": lat,
                "lon_centro": lon,
                "fecha": d,
                "n_harmonized": n_h,
                "phi_harmonized": phi_h,
            }
        )
    table = pa.Table.from_pylist(
        rows,
        schema=pa.schema(
            [
                ("cell_id", pa.int32()),
                ("lat_centro", pa.float32()),
                ("lon_centro", pa.float32()),
                ("fecha", pa.date32()),
                ("n_harmonized", pa.int32()),
                ("phi_harmonized", pa.float32()),
            ]
        ),
    )
    pq.write_table(table, path)


def _create_synthetic_weather(weather_dir: Path) -> None:
    weather_dir.mkdir(parents=True, exist_ok=True)
    # NASA POWER native grid: 0.5 deg lat x 0.625 deg lon
    nodes = [(-35.0, -66.25), (-33.0, -65.0)]
    rows = []
    for start_y, end_y in [(2023, 2023), (2024, 2024)]:
        d = date(start_y, 5, 1)
        end_d = date(end_y, 7, 1)
        while d <= end_d:
            for lat_c, lon_c in nodes:
                # Rain on June 10, dry afterward to test dry-spell streak
                precip = 5.0 if d.day == 10 else 0.0
                rows.append(
                    {
                        "lat_clima": lat_c,
                        "lon_clima": lon_c,
                        "elevacion_m": 450.0,
                        "fecha": d,
                        "temperature_2m_max": 28.0,
                        "relative_humidity_2m": 30.0,
                        "wind_speed_10m_max": 20.0,
                        "precipitation_sum": precip,
                        "shortwave_radiation_sum": 22.0,
                    }
                )
            d += timedelta(days=1)

    table = pa.Table.from_pylist(rows, schema=NASA_SCHEMA)
    pq.write_table(table, weather_dir / "nasapower_synth.parquet")


class PredictivePipelineTests(unittest.TestCase):
    def test_mediana_historica_mes_uses_only_prior_years(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            firms_path = tmp_path / "firms.parquet"
            weather_dir = tmp_path / "weather"
            _create_synthetic_firms(firms_path)
            _create_synthetic_weather(weather_dir)

            X_tr, y_tr, f_tr, X_va, y_va, f_va = build_training_dataset(
                firms_parquet=firms_path,
                weather_glob=(weather_dir / "*.parquet").as_posix(),
                neg_sample_permille=1000,
            )
            med = FEATURE_COLUMNS.index("mediana_historica_mes")
            lat = FEATURE_COLUMNS.index("lat_centro")

            # 2023 positives have no prior-year June fires: feature must not see their own label
            self.assertGreater(int((y_tr == 1).sum()), 0)
            self.assertTrue((X_tr[y_tr == 1, med] == 0.0).all())

            # 2024-06-20 rows only see 2023 June history
            day = f_va == f_va[0].__class__("2024-06-20")
            cell1 = day & (y_va == 1) & np.isclose(X_va[:, lat], -34.89)
            cell2 = day & (y_va == 1) & np.isclose(X_va[:, lat], -33.10)
            self.assertAlmostEqual(float(X_va[cell1][0, med]), 2.5, places=4)  # median(2, 3)
            self.assertAlmostEqual(float(X_va[cell2][0, med]), 1.0, places=4)  # excludes 2024-06-19

    def test_climate_node_mapping_and_no_leakage_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            firms_path = tmp_path / "firms.parquet"
            weather_dir = tmp_path / "weather"
            model_dir = tmp_path / "models"

            _create_synthetic_firms(firms_path)
            _create_synthetic_weather(weather_dir)

            for _, lat_min, lat_max, lon_min, lon_max in ARGENTINA_TILES:
                self.assertLessEqual(lat_max - lat_min, 10.0)
                self.assertLessEqual(lon_max - lon_min, 10.0)

            X_tr, y_tr, _, X_va, y_va, f_va = build_training_dataset(
                firms_parquet=firms_path,
                weather_glob=(weather_dir / "*.parquet").as_posix(),
                neg_sample_permille=500,
            )
            self.assertGreater(len(y_tr), 0)
            self.assertGreater(len(y_va), 0)

            # Verify June 15 (< 10 MW) is NOT labeled positive in validation
            june15_mask = f_va == f_va[0].__class__("2024-06-15")
            if june15_mask.any():
                self.assertEqual(int(y_va[june15_mask].sum()), 0)

            # Verify dry-spell streak on June 20 (last rain June 10 -> 10 dry days)
            june20_pos = (f_va == f_va[0].__class__("2024-06-20")) & (y_va == 1)
            self.assertTrue(june20_pos.any())
            dry_idx = FEATURE_COLUMNS.index("dias_consecutivos_sin_lluvia")
            self.assertAlmostEqual(float(X_va[june20_pos][0, dry_idx]), 10.0, places=3)

            # Verify end-to-end training & artifact export
            meta = train_and_export(
                firms_parquet=firms_path,
                weather_glob=(weather_dir / "*.parquet").as_posix(),
                model_dir=model_dir,
                neg_sample_permille=500,
            )
            model_file = model_dir / "fire_risk_lgbm.txt"
            meta_file = model_dir / "fire_risk_metadata.json"
            self.assertTrue(model_file.exists())
            self.assertTrue(meta_file.exists())

            saved_meta = json.loads(meta_file.read_text(encoding="utf-8"))
            self.assertEqual(saved_meta["features"], FEATURE_COLUMNS)
            self.assertIn("roc_auc", meta["validation_metrics"])

            booster = lgb.Booster(model_file=str(model_file))
            preds = booster.predict(X_va)
            self.assertTrue(((preds >= 0.0) & (preds <= 1.0)).all())


if __name__ == "__main__":
    unittest.main()
