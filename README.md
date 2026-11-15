# ModeloPredictivo — Riesgo Diario de Incendios (`lgbm-risk-24h-v2`)

Pipeline de entrenamiento, evaluación *out-of-time* y exportación de inferencia para predecir la probabilidad diaria real (próximas 24 h) de actividad crítica de incendios ($n_{\text{harm}} \ge 1 \wedge \Phi_{\text{harm}} \ge 10.0\text{ MW}$) sobre las **64.653 celdas de $5 \times 5\text{ km}$** de Argentina. En el holdout ciego de **2026** (`173.598` muestras), el modelo alcanza **$\text{ROC-AUC} = 0.9196$**, **$\text{PR-AUC}$ poblacional $= 0.0785$** ($3.12\times$ sobre el baseline de recencia), **calibración global $= 1.0417$** y **Lift $= 275.9\times$** en categoría `Extremo`.

> Para la formulación matemática completa y el estudio de ablación, ver [DOCUMENTACION_MODELO_PREDICTIVO.md](DOCUMENTACION_MODELO_PREDICTIVO.md).

---

## Cómo correrlo y replicarlo (Quick Path)

### 1. Crear el entorno e instalar dependencias

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install -r requirements.txt
```

> **Requisito de datos en el repo:** `firms_argentina_harmonized.parquet` (raíz del proyecto) y `data/nasapower_2018.parquet` .. `data/nasapower_2026.parquet`.

### 2. Verificar o descargar datos meteorológicos (NASA POWER / MERRA-2)

```powershell
# Verificar cobertura local 2018–2026
.\.venv\Scripts\python.exe ingest_nasapower.py --status

# (Opcional) Re-descargar desde NASA POWER por rango de años
.\.venv\Scripts\python.exe ingest_nasapower.py --start-year 2018 --end-year 2026
```

### 3. Entrenar, evaluar y exportar artefactos

```powershell
# Entrena Train (2018–2023) / Val (2024–2025) / Test (2026) -> models/fire_risk_lgbm.txt + fire_risk_metadata.json
.\.venv\Scripts\python.exe train_predictive_risk.py

# Evalúa sobre el holdout ciego 2026 (ROC-AUC, PR-AUC, calibración, Top-K y categorías)
.\.venv\Scripts\python.exe evaluate_model.py --test-start 2026-01-01 --test-end 2026-12-31

# Exporta bundle autocontenido (models/fire_risk_bundle.joblib) y tabla 7d (models/riesgo_celda_7d.parquet)
.\.venv\Scripts\python.exe export_backend.py --horizon-days 7
```

### 4. Ejecutar la suite de pruebas

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

---

## Arquitectura y Artefactos

| Componente | Rol | Salida / Efecto |
|---|---|---|
| `ingest_nasapower.py` | Descarga grilla MERRA-2 ($0.5^\circ \times 0.625^\circ$) en 12 teselas ($\le 8.5^\circ \times 7.0^\circ$) | `data/**/nasapower_2018..2026.parquet` |
| `train_predictive_risk.py` | Construye 20 features *leak-free* en DuckDB por lotes anuales, muestrea negativos ($8\text{‰}$), entrena LightGBM (`best_iteration = 494`) y aplica corrección de prior ($\Delta_{\text{logit}} = -4.828314$) | `models/fire_risk_lgbm.txt`, `models/fire_risk_metadata.json` |
| `evaluate_model.py` | Calcula métricas poblacionales ponderadas por Horvitz-Thompson ($w_0 = 125$) sobre el holdout 2026 | Reporte en consola |
| `export_backend.py` | Empaqueta `FireRiskPredictor` y precalcula riesgo multi-horizonte | `models/fire_risk_bundle.joblib`, `models/riesgo_celda_7d.parquet` |

---

## Pruebas Realizadas (Pipeline y Rendimiento)

La suite automatizada en `tests/` valida tanto la ausencia de fuga temporal como el desempeño estadístico del modelo entrenado:

| Archivo | Prueba | Qué verifica |
|---|---|---|
| `tests/test_predictive_pipeline.py` | `test_mediana_historica_mes_uses_only_prior_years` | `mediana_historica_mes` usa solo años previos (`< año(t)`) y excluye eventos del día anterior |
| `tests/test_predictive_pipeline.py` | `test_climate_node_mapping_and_no_leakage_training` | Teselas NASA POWER $\le 10^\circ$, corte $\Phi_{\text{harm}} \ge 10\text{ MW}$ y cálculo de racha seca |
| `tests/test_model_performance.py` | `test_prior_correction_undoes_sampling_and_weighting` | Exactitud analítica de $\Delta_{\text{logit}} = \ln(r) - \ln(w_{+})$ |
| `tests/test_model_performance.py` | `test_population_weights_and_percentile_cutoffs` | Pesos de Horvitz-Thompson ($w_0 = 1/r$) y cuantiles poblacionales ponderados |
| `tests/test_model_performance.py` | `test_real_world_calibration_with_negative_sampling` | Razón de calibración dentro de $[0.80, 1.25]$ bajo submuestreo negativo |
| `tests/test_model_performance.py` | `test_unseen_2026_holdout_performance` | Paridad con `evaluate_model.py` en holdout 2026 ($\text{ROC-AUC} \ge 0.85$, $\text{Brier} \le 0.20$) |
| `tests/test_model_performance.py` | `test_trained_model_on_real_2026_holdout` | Auditoría de `models/fire_risk_lgbm.txt` en 2026: $\text{PR-AUC} \ge \text{Baseline} + 0.05$, monotonicidad entre categorías y $\text{Lift}_{\text{Extremo}} \ge 20\times$ |

---

## Resultados Obtenidos (`models/fire_risk_metadata.json`)

### Métricas Globales (Ponderadas a Población Real)

| Métrica | Validación (`2024–2025`, $n=421.793$) | Holdout Ciego (`2026`, $n=173.598$) |
|---|---:|---:|
| Prevalencia real ($\pi$) | `0.000926` (`0.0926%`) | `0.000599` (`0.0599%`) |
| Probabilidad media predicha ($\bar{\hat{p}}$) | `0.000933` (`0.0933%`) | `0.000624` (`0.0624%`) |
| Razón de calibración ($\bar{\hat{p}} / \pi$) | `1.0072` | `1.0417` |
| **ROC-AUC** | `0.9284` | **`0.9196`** |
| **PR-AUC poblacional** | `0.0691` | **`0.0785`** |
| PR-AUC baseline (`anios_desde_ultimo_fuego`) | `0.0307` | `0.0252` (`3.12×` ganancia) |
| Brier Score | `0.000890` | `0.000576` |
| Log-Loss | `0.005435` | `0.003810` |

### Desempeño por Categoría Operativa (Holdout 2026)

| Categoría | Umbral ($\hat{p}_{\min}$ en Val) | % Celdas-Día (`2026`) | Precisión Real | Recall Individual | Lift sobre Azar |
|---|---:|---:|---:|---:|---:|
| `Extremo` | $\ge 0.057851$ (`5.79%`) | `0.0415%` | `16.53%` | `11.45%` | **`275.9×`** |
| `Alto` | $\ge 0.012920$ (`1.29%`) | `0.4062%` | `2.49%` | `16.88%` | **`41.6×`** |
| `Moderado` | $\ge 0.003550$ (`0.35%`) | `3.0420%` | `0.54%` | `27.52%` | **`9.0×`** |
| `Bajo` | $< 0.003550$ | `96.5103%` | `0.027%` | `44.15%` | `0.46×` |
