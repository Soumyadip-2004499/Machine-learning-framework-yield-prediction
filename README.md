# CropSense AI: AI-Powered Corn and Soybean Crop Yield Prediction System

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20XGBoost%20%7C%20LightGBM-F7931E.svg)](https://scikit-learn.org)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Production-Ready-brightgreen.svg)]()

> **CropSense AI** is a scientifically rigorous, data-driven, and farmer-centric crop yield forecasting and agricultural decision support system designed specifically for **Corn (*Zea mays*)** and **Soybean (*Glycine max*)**. Built upon 39 years of contiguous meteorological observations, multi-layer soil hydrology measurements, and weekly planting timelines across 105 Corn Belt counties, CropSense AI bridges advanced machine learning and practical field agronomics.

---

## Table of Contents
1. [Executive Summary & Core Objectives](#1-executive-summary--core-objectives)
2. [Research Foundations & Provenance](#2-research-foundations--provenance)
3. [Dataset Architecture & Auditing](#3-dataset-architecture--auditing)
4. [Data Cleaning, Anomaly Handling & Pruning](#4-data-cleaning-anomaly-handling--pruning)
5. [Machine Learning Benchmarks & Comparative Evaluation](#5-machine-learning-benchmarks--comparative-evaluation)
6. [Tuned Champion Models & Hyperparameters](#6-tuned-champion-models--hyperparameters)
7. [System Architecture & Full-Stack Implementation](#7-system-architecture--full-stack-implementation)
8. [Automated Suite of 7 Jupyter Notebooks](#8-automated-suite-of-7-jupyter-notebooks)
9. [REST API Documentation & Endpoints](#9-rest-api-documentation--endpoints)
10. [Farmer-Centric UI/UX & Interactive Capabilities](#10-farmer-centric-uiux--interactive-capabilities)
11. [Installation, Testing & Execution Guide](#11-installation-testing--execution-guide)
12. [Agronomic Takeaways & Future Directions](#12-agronomic-takeaways--future-directions)

---

## 1. Executive Summary & Core Objectives

### The Agricultural Problem
Crop yield prediction is crucial for global food security, grain commodity marketing, crop insurance underwriting, and farm-level input planning. However, conventional statistical yield models often fail to capture the **severe, non-linear biological penalties** caused by extreme climate shocks—such as reproductive heat desiccation during July pollination or late-summer vapor pressure deficits during soybean pod fill.

### What CropSense AI Delivers
- **100% Genuine Machine Learning Inference:** Zero placeholder predictions or hardcoded numbers. All estimates are generated directly by trained and serialized XGBoost gradient boosting pipelines.
- **Scientifically Grounded Feature Pruning:** Pruned 30+ zero-variance summer snow features while preserving critical multi-depth soil layers and weekly weather dynamics.
- **Exhaustive 16-Algorithm Benchmark:** Evaluated Linear, Regularized (Ridge, Lasso, ElasticNet), Tree Ensembles (Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting, AdaBoost), Kernel (SVR RBF), Distance (KNN), Gradient Boosting (XGBoost, LightGBM), and Meta-Ensembles (Voting, Stacking) on strict 5-fold cross-validation and untouched test splits.
- **Dual Prediction Paradigms:** 
  1. *Farmer-Friendly Mode:* 9 intuitive field sliders (rainfall, peak temperature, night temperature, soil texture, pH, organic matter, drainage, planting window, irrigation) calibrated to underlying multi-layer feature distributions.
  2. *Agronomist Advanced Mode:* Full 400-variable parameter vector interface.
- **Actionable Decision Support:** Automated Soil Health Index (0–100 score), benchmark yield comparisons against 39-year historical baselines, and plain-language agronomic advisories.
- **Global Unit Conversions:** Real-time conversion across US bushels per acre (`bu/acre`), International metric tons per hectare (`t/ha`), and South Asian quintals per acre (`qtl/acre`).

---

## 2. Research Foundations & Provenance

CropSense AI is built upon research methodologies and longitudinal datasets established in peer-reviewed agricultural data science:

1. **Primary Dataset Architecture:**
   - **Iniyan, S., et al. (2021).** *"Crop Yield Prediction using Machine Learning and Meteorological Variables."* *Wireless Personal Communications*, Springer.
   - **Data Sources:** 
     - **USDA-NASS:** County-level historical grain yield records (1980–2018).
     - **NASA Daymet v3:** 1 km x 1 km gridded daily weather aggregated into 52 weekly intervals ($W_1$ to $W_6$).
     - **USDA-NRCS gSSURGO:** Gridded Soil Survey Geographic database providing 10 soil physical and chemical parameters across 10 depth layers down to 200 cm ($S_1$ to $S_{10}$) plus 4 surface metrics ($S_{\text{surface}}$).
     - **USDA Crop Progress Reports:** Weekly state-level planting progress percentages ($P_1$ to $P_{16}$).
2. **Extreme Climate Modeling & Stacking Ensembles:**
   - **Waqar et al. (2025).** *"A Stacking Ensemble Framework Leveraging Synthetic Data for Accurate and Stable Crop Yield Forecasting."*
   - Documents non-linear threshold effects of reproductive heat desiccation ($>30^\circ\text{C}$) on corn silking and soybean pod enlargement.

---

## 3. Dataset Architecture & Auditing

| Parameter | Corn Dataset (`corn_data_csv.csv`) | Soybean Dataset (`Soybean_data_csv.csv`) |
|:---|:---:|:---:|
| **Total County-Year Observations** | 4,000 | 4,000 |
| **Raw Column Count** | 435 | 433 |
| **Number of US Counties Covered** | 105 Corn Belt Counties | 105 Corn Belt Counties |
| **Historical Period** | 1980 – 2018 (39 continuous years) | 1980 – 2018 (39 continuous years) |
| **Target Variable** | `Yield` (bushels / acre) | `Yield` (bushels / acre) |
| **Historical Mean Yield** | **137.38 bu/acre** ($8.62\text{ t/ha}$) | **42.49 bu/acre** ($2.86\text{ t/ha}$) |
| **Yield Standard Deviation** | 38.02 bu/acre | 10.14 bu/acre |
| **Historical Range** | 19.0 to 246.7 bu/acre | 13.0 to 82.3 bu/acre |
| **Missing Values in Raw CSV** | 0 (100% complete) | 0 (100% complete) |
| **Duplicate Observations** | 0 (Zero redundancy) | 0 (Zero redundancy) |
| **Features After Pruning** | **400 features** | **402 features** |

### Data Dictionary & Variable Hierarchy
- **Weather ($W_{1..6, 1..52}$):** 
  - $W_1$: Precipitation ($\text{mm/week}$)
  - $W_2$: Vapor Pressure Deficit ($\text{Pa}$)
  - $W_3$: Snow Water Equivalent ($\text{kg/m}^2$)
  - $W_4$: Maximum Daily Temperature ($\text{^\circ C}$)
  - $W_5$: Minimum Daily Temperature ($\text{^\circ C}$)
  - $W_6$: Incident Solar Radiation ($\text{W/m}^2$)
- **Soil Hydrology & Chemistry ($S_{1..10, 1..10}$):** 
  - 10 Depth increments: Layer 1 (0–5 cm), Layer 2 (5–15 cm), ..., Layer 10 (150–200 cm).
  - Properties: Bulk density ($S_1$), Cation Exchange Capacity ($S_2, S_9$), Clay content ($S_3$), Available water capacity ($S_4 - S_5$), pH ($S_{10}$), Organic Matter ($S_7$).
- **Surface Soil Index ($S_{\text{surface} 1..4}$):** Productivity index, drainage class, slope gradient.
- **Planting Progress ($P_{1..16}$):** Weekly cumulative planting percentage from April through June.

---

## 4. Data Cleaning, Anomaly Handling & Pruning

### 1. The Column 110 Header Anomaly (`'*'`)
During the raw corn data audit, column index 110 was identified with the corrupted header `'*'`. Surrounding columns were `W32`, `W33`, and `W35`. Meteorological sequence verification conclusively confirmed this column as **$W_{34}$ (Snow Water Equivalent, Week 4)**. The column was renamed and cast from string objects to standard `float64`.

### 2. Climatological Pruning of Summer Snow
In the Midwestern Corn Belt, weeks 16 through 45 span May through November. In this temperate climate, Snow Water Equivalent is physically 0.0 mm throughout the summer. Retaining 30+ zero-variance columns induces multicollinearity and wastes parameter capacity. These uninformative features ($W_{316}$–$W_{345}$) were safely pruned.

### 3. Target Leakage Prevention & Train/Test Isolation
All normalization scalers (`StandardScaler`) were wrapped inside scikit-learn `Pipeline` objects and fitted **strictly on training folds during cross-validation**, guaranteeing zero data leakage into validation or test sets.

---

## 5. Machine Learning Benchmarks & Comparative Evaluation

All models were evaluated on an untouched 20% holdout test set (800 records) and compared across 5-fold cross-validation on the 80% training set (3,200 records).

### Corn Regression Benchmark Table

| Model | Algorithmic Family | CV R² (Mean ± Std) | CV RMSE (bu/ac) | CV MAE (bu/ac) | Test R² | Test RMSE (bu/ac) | Test MAE (bu/ac) | Test MAPE (%) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Tuned XGBoost (Selected Final)** | **Boosting** | **0.9324 ± 0.0033** | **9.91 ± 0.28** | **7.49** | **0.9333** | **9.69** | **7.15** | **5.76%** |
| XGBoost (Default) | Boosting | 0.9270 ± 0.0031 | 10.29 ± 0.27 | 7.78 | 0.9328 | 9.73 | 7.24 | 5.91% |
| LightGBM | Boosting | 0.9260 ± 0.0052 | 10.36 ± 0.28 | 7.80 | 0.9315 | 9.82 | 7.32 | 5.99% |
| Stacking Regressor (RF + XGB + Ridge) | Meta-Ensemble | 0.9284 ± 0.0029 | 10.19 ± 0.15 | 7.76 | 0.9284 | 10.04 | 7.53 | 6.13% |
| HistGradientBoosting | Boosting | 0.9225 ± 0.0060 | 10.60 ± 0.37 | 8.02 | 0.9267 | 10.16 | 7.56 | 6.22% |
| Voting Ensemble (RF + XGB + LGB + ET) | Meta-Ensemble | 0.9221 ± 0.0034 | 10.63 ± 0.21 | 8.07 | 0.9259 | 10.22 | 7.64 | 6.28% |
| Linear Regression | Linear Baseline | 0.9253 ± 0.0036 | 10.41 ± 0.27 | 7.98 | 0.9256 | 10.24 | 7.85 | 6.37% |
| SVR (RBF Kernel) | Kernel | 0.9200 ± 0.0044 | 10.77 ± 0.33 | 7.99 | 0.9255 | 10.24 | 7.46 | 6.17% |
| Extra Trees | Bagged Trees | 0.9163 ± 0.0023 | 11.02 ± 0.16 | 8.36 | 0.9195 | 10.65 | 7.90 | 6.52% |
| Random Forest | Bagged Trees | 0.9026 ± 0.0056 | 11.89 ± 0.33 | 9.00 | 0.9104 | 11.23 | 8.37 | 6.89% |
| Ridge Regression ($\alpha=100$) | Regularized | 0.9009 ± 0.0042 | 11.99 ± 0.21 | 9.23 | 0.9073 | 11.42 | 8.67 | 7.12% |
| Lasso Regression ($\alpha=0.1$) | Regularized | 0.8977 ± 0.0037 | 12.19 ± 0.17 | 9.40 | 0.8977 | 12.00 | 9.25 | 7.55% |
| ElasticNet ($\alpha=0.1, l_1=0.5$) | Regularized | 0.8930 ± 0.0041 | 12.46 ± 0.21 | 9.61 | 0.8956 | 12.12 | 9.24 | 7.61% |
| Gradient Boosting (sklearn) | Boosting | 0.8788 ± 0.0060 | 13.26 ± 0.37 | 10.30 | 0.8768 | 13.17 | 10.02 | 8.27% |
| KNN Regressor ($k=7$) | Distance | 0.8729 ± 0.0102 | 13.57 ± 0.50 | 10.16 | 0.8721 | 13.42 | 10.17 | 8.39% |
| Decision Tree Regressor | Single Tree | 0.7794 ± 0.0193 | 17.88 ± 0.89 | 13.18 | 0.7837 | 17.45 | 12.89 | 10.69% |
| AdaBoost Regressor | Boosting | 0.6417 ± 0.0283 | 22.79 ± 0.88 | 18.39 | 0.6527 | 22.12 | 17.94 | 14.83% |

---

### Soybean Regression Benchmark Table

| Model | Algorithmic Family | CV R² (Mean ± Std) | CV RMSE (bu/ac) | CV MAE (bu/ac) | Test R² | Test RMSE (bu/ac) | Test MAE (bu/ac) | Test MAPE (%) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Tuned XGBoost (Selected Final)** | **Boosting** | **0.9146 ± 0.0063** | **2.95 ± 0.10** | **2.24** | **0.9241** | **2.82** | **2.14** | **5.53%** |
| SVR (RBF Kernel) | Kernel | 0.9046 ± 0.0104 | 3.11 ± 0.15 | 2.31 | 0.9203 | 2.89 | 2.18 | 5.56% |
| XGBoost (Default) | Boosting | 0.9102 ± 0.0057 | 3.02 ± 0.08 | 2.30 | 0.9171 | 2.95 | 2.25 | 5.85% |
| LightGBM | Boosting | 0.9116 ± 0.0044 | 3.00 ± 0.06 | 2.27 | 0.9129 | 3.02 | 2.28 | 5.89% |
| Stacking Regressor (RF + XGB + Ridge) | Meta-Ensemble | 0.9090 ± 0.0061 | 3.04 ± 0.06 | 2.33 | 0.9099 | 3.08 | 2.35 | 5.99% |
| HistGradientBoosting | Boosting | 0.9065 ± 0.0033 | 3.09 ± 0.06 | 2.34 | 0.9084 | 3.10 | 2.35 | 6.08% |
| Linear Regression | Linear Baseline | 0.9110 ± 0.0045 | 3.01 ± 0.03 | 2.34 | 0.9080 | 3.11 | 2.33 | 5.94% |
| Voting Ensemble (RF + XGB + LGB + ET) | Meta-Ensemble | 0.9001 ± 0.0058 | 3.19 ± 0.08 | 2.42 | 0.9048 | 3.16 | 2.41 | 6.27% |
| Extra Trees | Bagged Trees | 0.8915 ± 0.0086 | 3.32 ± 0.12 | 2.53 | 0.9006 | 3.23 | 2.48 | 6.43% |
| Random Forest | Bagged Trees | 0.8726 ± 0.0058 | 3.60 ± 0.08 | 2.73 | 0.8755 | 3.62 | 2.73 | 7.11% |
| Ridge Regression ($\alpha=100$) | Regularized | 0.8790 ± 0.0080 | 3.51 ± 0.06 | 2.71 | 0.8745 | 3.63 | 2.80 | 7.23% |
| Gradient Boosting (sklearn) | Boosting | 0.8560 ± 0.0037 | 3.83 ± 0.06 | 2.95 | 0.8517 | 3.95 | 3.06 | 8.05% |
| KNN Regressor ($k=7$) | Distance | 0.8349 ± 0.0164 | 4.09 ± 0.17 | 3.07 | 0.8483 | 3.99 | 3.02 | 7.88% |
| ElasticNet ($\alpha=0.1, l_1=0.5$) | Regularized | 0.8541 ± 0.0098 | 3.85 ± 0.08 | 2.96 | 0.8437 | 4.05 | 3.16 | 8.28% |
| Lasso Regression ($\alpha=0.1$) | Regularized | 0.8446 ± 0.0115 | 3.98 ± 0.10 | 3.05 | 0.8328 | 4.19 | 3.26 | 8.59% |
| Decision Tree Regressor | Single Tree | 0.7218 ± 0.0086 | 5.33 ± 0.14 | 3.97 | 0.7617 | 5.00 | 3.68 | 9.36% |
| AdaBoost Regressor | Boosting | 0.5920 ± 0.0252 | 6.45 ± 0.24 | 5.06 | 0.5794 | 6.65 | 5.22 | 13.98% |

---

## 6. Tuned Champion Models & Hyperparameters

Both crop champions utilize **XGBoost Regressor**, fine-tuned through 3-fold cross-validation grid search optimizing RMSE:

```python
# Optimal Hyperparameter Configuration
champion_hyperparameters = {
    "colsample_bytree": 0.85,    # 85% feature subsampling prevents single-feature dominance
    "learning_rate": 0.07,       # Controlled shrinkage prevents gradient overshoot
    "max_depth": 5,              # 5-level interaction depth captures weather-soil thresholds
    "n_estimators": 300,         # 300 boosting rounds with early stabilization
    "subsample": 0.85,           # 85% sample subsampling for variance reduction
    "objective": "reg:squarederror"
}
```

### Production Artifact Storage
- **Corn Champion:** `backend/models/corn_model_final.pkl` (785 KB)
- **Soybean Champion:** `backend/models/soybean_model_final.pkl` (784 KB)
- **Payload Contents:** Trained model object, feature column schema, precomputed training feature distribution metrics (mean/std), historical yield statistics, and cross-validation performance metrics.

---

## 7. System Architecture & Full-Stack Implementation

```
CropSense AI/
├── backend/                        # Python FastAPI Backend
│   ├── app/
│   │   ├── main.py                 # Application factory & static mounting
│   │   ├── routes/
│   │   │   ├── predict.py          # /api/predict and /api/predict/farmer-simple
│   │   │   └── health.py           # /api/health and /api/benchmarks
│   │   ├── schemas/
│   │   │   └── prediction.py       # Pydantic v2 validation contracts
│   │   └── services/
│   │       └── prediction_service.py # Model loading, feature mapping, advisory engine
│   ├── models/                     # Production model artifacts
│   │   ├── corn_model_final.pkl
│   │   ├── soybean_model_final.pkl
│   │   └── run_model_experiments.py
│   └── tests/
│       └── test_api.py             # 100% passing pytest automated test suite
├── frontend/                       # Interactive User Interface
│   └── public/
│       ├── index.html              # Semantic HTML5 layout
│       ├── styles.css              # Custom Farmland theme + Radial circular reveal
│       └── app.js                  # Client logic, unit switcher, reactive UI
├── notebooks/                      # 7 Executable Jupyter Notebooks
│   ├── 01_corn_eda_and_data_cleaning.ipynb
│   ├── 02_corn_regression_comparison.ipynb
│   ├── 03_corn_hyperparameter_tuning.ipynb
│   ├── 04_soybean_eda_and_data_cleaning.ipynb
│   ├── 05_soybean_regression_comparison.ipynb
│   ├── 06_soybean_hyperparameter_tuning.ipynb
│   └── 07_final_model_evaluation.ipynb
├── data/
│   ├── raw/                        # Original USDA / Iniyan et al. CSV datasets
│   └── processed/                  # Cleaned and pruned datasets
└── reports/
    ├── figures/                    # High-resolution publication charts
    └── model_comparison/           # Markdown and CSV comparison tables
```

---

## 8. Automated Suite of 7 Jupyter Notebooks

Located in `notebooks/`, each notebook contains complete markdown documentation, executable Python cells, and embedded high-resolution figures:

1. **`01_corn_eda_and_data_cleaning.ipynb`** (3.9 MB, 16 cells)
   - Research context (Iniyan et al. 2021), column `'*'` resolution, zero duplicate validation, summer snow pruning, yield distribution, July heat stress scatter, top correlation heatmap.
2. **`02_corn_regression_comparison.ipynb`** (932 KB, 9 cells)
   - 16 regression algorithms benchmarked across 5-fold CV with leakage-free pipelines, comparative bar charts, residual diagnostics, agronomic performance takeaways.
3. **`03_corn_hyperparameter_tuning.ipynb`** (6.8 KB, 8 cells)
   - XGBoost GridSearchCV parameter search, best parameter analysis, test set validation, top-15 Gini feature importances, artifact serialization.
4. **`04_soybean_eda_and_data_cleaning.ipynb`** (707 KB, 12 cells)
   - Soybean biology, biological nitrogen fixation, August pod-filling water stress, target yield distribution, correlation heatmaps, pruned dataset export.
5. **`05_soybean_regression_comparison.ipynb`** (917 KB, 7 cells)
   - 16 regression algorithms benchmarked on soybean, model comparison tables, test set bar chart, residual error analysis.
6. **`06_soybean_hyperparameter_tuning.ipynb`** (5.7 KB, 6 cells)
   - XGBoost grid search optimization, error reporting in bu/acre, t/ha, and qtl/acre, feature gain rankings.
7. **`07_final_model_evaluation.ipynb`** (10.2 KB, 9 cells)
   - Head-to-head cross-crop synthesis, simulated extreme drought and heat stress tests, 95% empirical prediction intervals, soil health engine validation.

---

## 9. REST API Documentation & Endpoints

The FastAPI server provides automated OpenAPI documentation at `http://localhost:8000/docs`:

### `POST /api/predict/farmer-simple`
Primary farmer endpoint accepting 9 intuitive field inputs:
```json
{
  "crop": "corn",
  "region": "Central Corn Belt (Iowa/Illinois)",
  "rainfall_seasonal_mm": 550.0,
  "temp_peak_c": 28.5,
  "temp_night_c": 17.0,
  "soil_texture": "Silt Loam",
  "soil_ph": 6.5,
  "organic_matter_pct": 3.5,
  "soil_drainage": "Well-drained",
  "planting_window": "Optimal (Late April - Early May)",
  "irrigation": false
}
```
**Sample Response:**
```json
{
  "success": true,
  "crop": "Corn",
  "predicted_yield": 156.4,
  "unit": "bu/acre",
  "predicted_yield_kg_ha": 9817.2,
  "predicted_yield_t_ha": 9.82,
  "predicted_yield_quintal_acre": 39.73,
  "confidence_interval_lower": 137.4,
  "confidence_interval_upper": 175.4,
  "soil_health": {
    "score": 90.0,
    "rating": "Excellent",
    "limiting_factors": []
  },
  "benchmark_comparison": {
    "historical_mean_bu_acre": 137.38,
    "delta_bu_acre": 19.02,
    "percent_difference": 13.85,
    "assessment": "Above Average"
  },
  "agronomic_advisories": [
    "Your predicted yield is 13.8% above historical county averages.",
    "Optimal soil organic matter (3.5%) is providing strong nutrient retention.",
    "Well-drained silt loam provides maximum available water capacity."
  ]
}
```

### Other Core Endpoints
- `GET /api/health`: Server health check, uptime, loaded models, memory status.
- `GET /api/benchmarks`: 39-year historical baselines and extrema for both crops.
- `GET /api/metadata`: Loaded model architecture, versions, hyperparameters, and feature counts.
- `POST /api/predict`: Advanced endpoint accepting raw 400-feature vectors.

---

## 10. Farmer-Centric UI/UX & Interactive Capabilities

The web interface is hosted directly from `frontend/public/` with a deeply agricultural design system:
- **Botanical Vine Scroll Progress Indicator:** A fluid organic vine crawls across the top of the viewport with a rotating leaf indicator tracking scroll position.
- **Crop-Reactive Ambient Breeze Particle Canvas:** 60/120 FPS GPU-composited canvas simulating natural field winds, floating golden corn pollen/chaff or emerald soybean leaflets depending on selected crop.
- **Living Crop Seedling & Root System SVG:** Dynamic botanical plant simulation showing taproot and lateral root penetration into multi-depth soil strata according to organic matter and soil health scores.
- **Interactive 3-Tier Soil Horizon Strata Explorer:** Clickable interactive layers for Horizon O/A (Topsoil 0–15 cm), Horizon B (Root Zone 15–60 cm), and Horizon C (Subsoil 60–150 cm) revealing USDA gSSURGO bulk density, available water capacity (AWC), and hydraulic conductivity.
- **Tactile Sliders with Live Agronomic Interpretation Badges:** Real-time feedback chips explaining physiological status as sliders move (e.g., *Optimal Silking*, *Severe Silk Desiccation*, *Prime Bioavailability*, *Exceptional Water Sponge*).
- **Radial Harvest Yield Gauge & Spring Counter:** Vintage agricultural dial sweep animation with spring-damped number count-up.
- **Dual-Mode Control:** Seamlessly switch between Farmer Mode (calibrated sliders) and Agronomist Mode.
- **Natural Farmland Theme & Dark Soil Theme:** Includes a **smooth radial circular reveal transition** centered on the user's click coordinates.
- **Three-Way Unit Switcher:** Toggle between US `bu/acre`, International `t/ha`, and South Asian `qtl/acre`.
- **Preconfigured Scenarios:** One-click presets for *Normal Season*, *Severe Drought (2012)*, and *Bountiful Year*.
- **Interactive Model Comparison:** Dynamic tab switching between Corn and Soybean benchmark tables with highlighted champion models.

---

## 11. Installation, Testing & Execution Guide

### Prerequisites
- Python 3.10+ (Tested on Python 3.13)
- Modern web browser (Chrome, Firefox, Edge, Safari)

### 1. Install Dependencies
```bash
pip install fastapi uvicorn xgboost lightgbm scikit-learn seaborn pandas numpy matplotlib tabulate nbformat
```

### 2. Run Automated API Tests
```bash
python -m pytest backend/tests/test_api.py -v
```
*Expected: 9 passed in ~3 seconds.*

### 3. Launch CropSense AI Application
From the repository root:
```bash
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```
Open **`http://localhost:8000`** in your browser. The web frontend and prediction backend run concurrently on the same port!

---

## 12. Agronomic Takeaways & Future Directions

1. **Mid-Summer Temperature Dominance:** Reproductive heat stress ($W4_{28..32}$ for corn, $W4_{32..35}$ for soybean) is the single most critical environmental determinant of yield decline.
2. **Topsoil Organic Matter Buffering:** Soil organic matter ($S_{7,1}$) directly dampens drought severity by maintaining available water capacity ($S_4 - S_5$) during dry intervals.
3. **Future Expansions:** Integrating satellite NDVI vegetative indices, field-level microclimate sensor feeds, and multi-crop rotation optimization (e.g., wheat, sorghum, and pulses).

---

## Author & Credits

- **Developer & Researcher:** Soumyadip Chattopadhyay
- **Platform:** CropSense AI
- **GitHub:** [@Soumyadip-2004499](https://github.com/Soumyadip-2004499)

---

*CropSense AI System — Engineered for Agricultural Excellence and Global Food Security.*
