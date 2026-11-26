"""Evaluate trained 24h fire-risk LightGBM model on the unseen 2026 holdout."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb

from modelo_predictivo.train_predictive_risk import (
    build_dataset_splits,
    evaluate_predictions,
    population_weights,
    predict_real_probability,
    recency_baseline_pr_auc,
)


def load_risk_cutoffs(metadata: dict[str, Any]) -> dict[str, float]:
    """Read per-category minimum probabilities from model metadata."""
    cats = metadata["risk_categories"]
    return {name: float(cats[name]["min_probability"]) for name in ("Extremo", "Alto", "Moderado")}


def evaluate_holdout(
    model_path: Path,
    firms_parquet: Path,
    weather_glob: str,
    metadata_path: Path | None = None,
    test_start: str = "2026-01-01",
    test_end: str = "2026-12-31",
    neg_sample_permille: int = 8,
) -> dict[str, Any]:
    """Load saved booster + metadata and evaluate real-world metrics on the holdout."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    metadata_path = metadata_path or model_path.with_name("fire_risk_metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    logit_offset = float(metadata["probability"]["logit_offset"])
    cutoffs = load_risk_cutoffs(metadata)

    target_years = list(range(int(test_start[:4]), int(test_end[:4]) + 1))
    _, _, _, _, _, _, X_test, y_test, _ = build_dataset_splits(
        firms_parquet=firms_parquet,
        weather_glob=weather_glob,
        test_start=test_start,
        test_end=test_end,
        neg_sample_permille=neg_sample_permille,
        only_years=target_years,
    )
    if len(y_test) == 0:
        raise RuntimeError(f"No holdout samples found for {test_start}..{test_end}.")

    booster = lgb.Booster(model_file=str(model_path))
    t0 = time.perf_counter()
    y_prob = predict_real_probability(booster, X_test, logit_offset)
    infer_s = time.perf_counter() - t0

    weights = population_weights(y_test, neg_sample_permille)
    metrics = evaluate_predictions(y_test, y_prob, weights, cutoffs)
    metrics["baseline_recency_pr_auc"] = recency_baseline_pr_auc(X_test, y_test, weights)
    metrics["holdout_period"] = f"{test_start}..{test_end}"
    metrics["inference_seconds"] = round(infer_s, 4)
    metrics["rows_per_second"] = round(len(y_test) / max(infer_s, 1e-6), 1)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate saved fire-risk model on unseen 2026 holdout data."
    )
    parser.add_argument("--model-path", type=Path, default=Path("models/fire_risk_lgbm.txt"))
    parser.add_argument("--metadata-path", type=Path, default=Path("models/fire_risk_metadata.json"))
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
    parser.add_argument("--test-start", type=str, default="2026-01-01")
    parser.add_argument("--test-end", type=str, default="2026-12-31")
    parser.add_argument("--neg-sample-permille", type=int, default=8)
    args = parser.parse_args()

    report = evaluate_holdout(
        model_path=args.model_path,
        firms_parquet=args.firms,
        weather_glob=args.weather_glob,
        metadata_path=args.metadata_path,
        test_start=args.test_start,
        test_end=args.test_end,
        neg_sample_permille=args.neg_sample_permille,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
