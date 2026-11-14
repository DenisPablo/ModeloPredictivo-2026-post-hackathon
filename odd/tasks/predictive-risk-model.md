# Predictive Fire Risk Model (Argentina, 24h Horizon)
Feature: LightGBM 24h fire-risk model (`lgbm-risk-24h-v2`) combining `firms_argentina_harmonized.parquet` and NASA POWER MERRA-2 daily weather (`data/weather/nasapower/nasapower_*.parquet`).

## Specs
- S1: Positive label `y = 1` for `t+1` (next 24h) is defined as `n_harmonized >= 1 AND phi_harmonized >= 10.0` from `firms_argentina_harmonized.parquet`.
- S2: Spatial scope is all of Argentina (`64,653` active 5x5 km cells mapped in O(1) to the native `0.5 deg lat x 0.625 deg lon` NASA POWER MERRA-2 grid across 12 regional tiles `<= 10x10 deg` via `ROUND(lat_centro * 2.0) / 2.0`, `ROUND(lon_centro / 0.625) * 0.625`).
- S3: Temporal scope is `2018-01-01` to `2026-12-31` with strict Time-Series Split (`train = 2018-01-01..2023-12-31`, `val = 2024-01-01..2025-12-31`, unseen `test = 2026-01-01..2026-12-31`) and deterministic negative subsampling (`neg_sample_permille = 8`).
- S4: Features must avoid future and same-year target leakage: fire memory (`focos_ultimos_7d`, `focos_ultimos_30d`, `frp_ultimos_7d`, `anios_desde_ultimo_fuego`) computed strictly prior to `t` (`< t`), and `mediana_historica_mes` computed with an expanding window over strictly prior years (`fecha < YYYY-01-01`); weather, topography, and drought features (`elevacion_m`, `temperature_2m_max`, `relative_humidity_2m`, `wind_speed_10m_max`, `precipitation_sum`, `shortwave_radiation_sum`, `lluvia_acum_7d`, `lluvia_acum_15d`, `lluvia_acum_30d`, `dias_consecutivos_sin_lluvia`, `vpd_approx`).
- S5: Deliverable is the trained LightGBM model artifact (`models/fire_risk_lgbm.txt`), metadata/metrics JSON (`models/fire_risk_metadata.json`) with Bayesian prior correction (`logit_offset = -4.828314`) and validation-derived percentile risk categories, standalone holdout evaluator (`evaluate_model.py`), automated test suite (`7 passed`), and paper-grade documentation (`DOCUMENTACION_MODELO_PREDICTIVO.md`).

## Tasks
- [x] T1 (S2, S3): Implement `ingest_nasapower.py` with 12 regional tiles (`8.5x7.0 deg <= 10 deg`) and parallel downloads (`ThreadPoolExecutor`), fetching 2018-2026 NASA POWER weather (`9,517,076` rows) | route: inline | commit: c242813, 2734b90
- [x] T2 (S1, S2, S3, S4, S5): Implement `train_predictive_risk.py` and `evaluate_model.py` (DuckDB yearly batches with 2GB limit, expanding-window `mediana_historica_mes`, Bayesian prior correction, population-weighted metrics, percentile risk categories, and 2026 holdout evaluation: `ROC-AUC = 0.9196`, `calibration_ratio = 1.0417`, `Lift_Extremo = 275.9x`) | route: inline | commit: 2734b90
- [x] T3 (S1-S5): Add automated unit, leakage, calibration, and holdout performance tests in `tests/test_predictive_pipeline.py` and `tests/test_model_performance.py` (`7 passed in 8.44s`) | route: inline | commit: 2734b90
- [x] T4 (S1-S5): Document the mathematical formulation, ablations, technical justifications, and test matrix in `DOCUMENTACION_MODELO_PREDICTIVO.md` | route: inline | commit: HEAD

## Log
- L1: "Nosotros lo que a hacer es construir un modelo predictivo usando firms_argentina_harmonized.parquet Y openmeteo. Vamos a iniciar con un SDD hazme todas las preguntas necesarias. La idea es que apartir de estos datos que meciono mas informacion consultada por OpenMeteo podamos crear un proyecto predictivo que sea capas de inferir informacion nueva, lo que sera un cronjob en el backend que consultara al modelo cada 1hs y marca posibles puntos calientes."
- L2: Target horizon: 24h ("Para las 24hs estaria bien, aunque luego en el frontend debe poderse seleccionar una ventana de tiempo donde se quiere filtrar, el backend no esta en nuestro scope").
- L3: Positive label: "Para la etiqueta positiva y = 1 en las próximas 24 h usá el criterio combinado n_harmonized >= 1 AND phi_harmonized >= 10.0 (en línea con la especificación RS-ARG-FUEGO-2026-V1)."
- L4: Spatial scope: "Arranquemos con la argentina" (64,653 cells).
- L5: Weather source migrated to NASA POWER regional daily MERRA-2 (`ingest_nasapower.py`, 12 tiles `8.5x7.0 deg`, `0.5x0.625 deg` grid, `2018-2026`).
- L6: Model effectiveness audit & v2 fix ("Si arreglalo y testealo luego dime el rendimiento del modelo y si es bueno"): fixed `mediana_historica_mes` self-label leakage (`fecha < YYYY-01-01`), added Bayesian prior correction (`logit_offset = -4.828314`) and Horvitz-Thompson population weights (`w_0 = 125`), and verified `ROC-AUC = 0.9196` and `calibration_ratio = 1.0417` on the unseen 2026 holdout (`7 passed`).
- L7: Paper-grade documentation and test/justification registry added in `DOCUMENTACION_MODELO_PREDICTIVO.md`.
