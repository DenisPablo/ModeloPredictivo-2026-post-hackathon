"""Automated performance & out-of-time (2026) holdout tests for the fire-risk model."""

from __future__ import annotations

import json
import math
import tempfile
import time
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from evaluate_model import evaluate_holdout
from ingest_nasapower import NASA_SCHEMA
from train_predictive_risk import (
    population_weights,
    prior_logit_offset,
    to_real_probability,
    train_and_export,
    weighted_top_cutoff,
)

REAL_MODEL = Path("models/fire_risk_lgbm.txt")
REAL_FIRMS = Path("firms_argentina_harmonized.parquet")
REAL_WEATHER_2026 = Path("data/weather/nasapower/nasapower_2026.parquet")


def _create_signal_dataset(firms_path: Path, weather_dir: Path) -> None:
    """Create synthetic 2022-2026 dataset with realistic fire-weather signal."""
    weather_dir.mkdir(parents=True, exist_ok=True)
    cells = [
        (1, -35.0, -66.25),
        (2, -33.0, -65.0),
        (3, -31.0, -63.75),
        (4, -29.0, -62.5),
    ]

    firms_rows: list[dict] = []
    weather_rows: list[dict] = []

    d = date(2022, 1, 1)
    end_d = date(2026, 6, 30)
    while d <= end_d:
        # High fire-risk regime on days 15..22 of each month (hot, dry, windy)
        is_hot_dry = 15 <= d.day <= 22
        is_rain = d.day in (1, 5, 10)
        for cid, lat_c, lon_c in cells:
            weather_rows.append(
                {
                    "lat_clima": lat_c,
                    "lon_clima": lon_c,
                    "elevacion_m": 300.0 + cid * 50.0,
                    "fecha": d,
                    "temperature_2m_max": 36.0 if is_hot_dry else 18.0,
                    "relative_humidity_2m": 18.0 if is_hot_dry else 75.0,
                    "wind_speed_10m_max": 32.0 if is_hot_dry else 10.0,
                    "precipitation_sum": 12.0 if is_rain else 0.0,
                    "shortwave_radiation_sum": 26.0 if is_hot_dry else 12.0,
                }
            )
            # Fire occurs during hot-dry spells for cells 1..3 (and prior day memory)
            if is_hot_dry and (d.day % 2 == 0) and cid <= 3:
                firms_rows.append(
                    {
                        "cell_id": cid,
                        "lat_centro": lat_c,
                        "lon_centro": lon_c,
                        "fecha": d,
                        "n_harmonized": 3,
                        "phi_harmonized": 45.0,
                    }
                )
        d += timedelta(days=1)

    firms_table = pa.Table.from_pylist(
        firms_rows,
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
    pq.write_table(firms_table, firms_path)
    pq.write_table(
        pa.Table.from_pylist(weather_rows, schema=NASA_SCHEMA),
        weather_dir / "nasapower_synth.parquet",
    )


class ModelPerformanceTests(unittest.TestCase):
    def test_unseen_2026_holdout_performance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            firms_path = tmp_path / "firms.parquet"
            weather_dir = tmp_path / "weather"
            model_dir = tmp_path / "models"

            _create_signal_dataset(firms_path, weather_dir)

            t0 = time.perf_counter()
            meta = train_and_export(
                firms_parquet=firms_path,
                weather_glob=(weather_dir / "*.parquet").as_posix(),
                model_dir=model_dir,
                neg_sample_permille=1000,
                test_start="2026-01-01",
                test_end="2026-12-31",
            )
            elapsed_s = time.perf_counter() - t0

            self.assertLess(elapsed_s, 60.0)
            self.assertIsNotNone(meta["test_metrics"])
            self.assertGreater(meta["split"]["test_rows"], 0)

            test_m = meta["test_metrics"]
            self.assertGreaterEqual(test_m["roc_auc"], 0.85)
            self.assertGreaterEqual(test_m["pr_auc"], 0.80)
            self.assertLessEqual(test_m["brier_score"], 0.20)

            saved = json.loads((model_dir / "fire_risk_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["split"]["test"], "2026-01-01..2026-12-31")
            self.assertEqual(saved["test_metrics"]["n_samples"], test_m["n_samples"])

            # Verify standalone holdout evaluator matches train-time 2026 metrics
            standalone = evaluate_holdout(
                model_path=model_dir / "fire_risk_lgbm.txt",
                firms_parquet=firms_path,
                weather_glob=(weather_dir / "*.parquet").as_posix(),
                neg_sample_permille=1000,
            )
            self.assertAlmostEqual(standalone["roc_auc"], test_m["roc_auc"], places=5)
            self.assertAlmostEqual(standalone["pr_auc"], test_m["pr_auc"], places=5)

    @unittest.skipUnless(
        REAL_MODEL.exists() and REAL_FIRMS.exists() and REAL_WEATHER_2026.exists(),
        "Trained model or real 2026 data not present yet.",
    )
    def test_trained_model_on_real_2026_holdout(self) -> None:
        report = evaluate_holdout(
            model_path=REAL_MODEL,
            firms_parquet=REAL_FIRMS,
            weather_glob="data/weather/nasapower/nasapower_*.parquet",
        )
        self.assertGreater(report["n_samples"], 1000)
        self.assertGreaterEqual(report["roc_auc"], 0.85)
        # Must clearly beat the strongest single-feature baseline (recency of last fire)
        self.assertGreaterEqual(report["pr_auc"], report["baseline_recency_pr_auc"] + 0.05)
        # Real-world calibration-in-the-large (tolerates year-to-year prevalence swings)
        self.assertGreaterEqual(report["calibration_ratio"], 0.5)
        self.assertLessEqual(report["calibration_ratio"], 2.0)
        # Categories must be ordered by real precision and Extremo must be highly selective
        cats = report["categories"]
        self.assertGreater(cats["Extremo"]["precision"], cats["Alto"]["precision"])
        self.assertGreater(cats["Alto"]["precision"], cats["Moderado"]["precision"])
        self.assertGreater(cats["Moderado"]["precision"], cats["Bajo"]["precision"])
        self.assertGreaterEqual(cats["Extremo"]["lift"], 20.0)


class ProbabilityCorrectionTests(unittest.TestCase):
    def test_prior_correction_undoes_sampling_and_weighting(self) -> None:
        offset = prior_logit_offset(neg_sample_permille=8, scale_pos_weight=2.0)
        self.assertAlmostEqual(offset, math.log(0.008) - math.log(2.0), places=12)
        # Margin 0 means 50% in the sampled/weighted space -> odds scaled by 0.008 / 2.0
        p = float(to_real_probability(np.array([0.0]), offset)[0])
        self.assertAlmostEqual(p, 0.004 / 1.004, places=9)

    def test_population_weights_and_percentile_cutoffs(self) -> None:
        y = np.array([1, 0, 0, 0])
        w = population_weights(y, neg_sample_permille=250)
        self.assertEqual(w.tolist(), [1.0, 4.0, 4.0, 4.0])
        prob = np.array([0.9, 0.5, 0.2, 0.1])
        # Total weight 13 (cumulative 1, 5, 9, 13): the top 0.5 units are the positive row
        self.assertEqual(weighted_top_cutoff(prob, w, 0.5 / 13), 0.9)
        self.assertEqual(weighted_top_cutoff(prob, w, 3 / 13), 0.5)

    def test_real_world_calibration_with_negative_sampling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            firms_path = tmp_path / "firms.parquet"
            weather_dir = tmp_path / "weather"
            _create_signal_dataset(firms_path, weather_dir)

            meta = train_and_export(
                firms_parquet=firms_path,
                weather_glob=(weather_dir / "*.parquet").as_posix(),
                model_dir=tmp_path / "models",
                neg_sample_permille=200,
            )
            # Probabilities are re-expanded to the full population despite 20% sampling
            ratio = meta["validation_metrics"]["calibration_ratio"]
            self.assertGreaterEqual(ratio, 0.8)
            self.assertLessEqual(ratio, 1.25)
            self.assertAlmostEqual(
                meta["probability"]["logit_offset"], math.log(0.2), places=9
            )


if __name__ == "__main__":
    unittest.main()
