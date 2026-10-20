"""
KrishiYield — End-to-End Data Preprocessing, Benchmarking & Hyperparameter Tuning Pipeline
Based on Iniyan & Jebakumar (2021) and Chattopadhyay, Sinha, & Iniyan S.
"""

import os
import sys
import json
import joblib
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score,
    root_mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)

# Linear models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
# Tree models
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    AdaBoostRegressor,
    VotingRegressor,
    StackingRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# Boosting models
import xgboost as xgb
import lightgbm as lgb

PROJ_DIR = r"C:\Users\chatt\.gemini\antigravity\scratch\krishiyield"
RAW_DIR = os.path.join(PROJ_DIR, "data", "raw")
PROC_DIR = os.path.join(PROJ_DIR, "data", "processed")
MODEL_DIR = os.path.join(PROJ_DIR, "backend", "models")
REPORT_DIR = os.path.join(PROJ_DIR, "reports", "model_comparison")
FIG_DIR = os.path.join(PROJ_DIR, "reports", "figures")

os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

RANDOM_STATE = 42

def save_markdown_table(df, file_path):
    try:
        df.to_markdown(file_path, index=False)
    except Exception:
        headers = list(df.columns)
        lines = []
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for _, row in df.iterrows():
            lines.append("| " + " | ".join([str(val) for val in row.values]) + " |")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

def preprocess_dataset(raw_filename, crop_name):
    print(f"\n==========================================")
    print(f"   PREPROCESSING {crop_name.upper()} RAW DATASET")
    print(f"==========================================")
    raw_path = os.path.join(RAW_DIR, raw_filename)
    df = pd.read_csv(raw_path)
    
    # Standardize column names (remove spaces, strip)
    df.columns = [c.strip() for c in df.columns]
    
    # Resolve any wildcard or unnamed columns
    if '*' in df.columns:
        df = df.drop(columns=['*'])
    unnamed = [c for c in df.columns if 'Unnamed' in c]
    if unnamed:
        df = df.drop(columns=unnamed)
        
    print(f"Initial shape: {df.shape}")
    print(f"Missing values: {df.isna().sum().sum()}")
    print(f"Duplicates: {df.duplicated().sum()}")
    
    # Save metadata summary
    meta_df = pd.DataFrame({
        'column': df.columns,
        'dtype': df.dtypes.astype(str),
        'missing': df.isna().sum().values,
        'unique_values': df.nunique().values,
        'mean': [df[c].mean() if np.issubdtype(df[c].dtype, np.number) else np.nan for c in df.columns],
        'std': [df[c].std() if np.issubdtype(df[c].dtype, np.number) else np.nan for c in df.columns],
        'min': [df[c].min() if np.issubdtype(df[c].dtype, np.number) else np.nan for c in df.columns],
        'max': [df[c].max() if np.issubdtype(df[c].dtype, np.number) else np.nan for c in df.columns]
    })
    meta_path = os.path.join(PROC_DIR, f"{crop_name}_meta.csv")
    meta_df.to_csv(meta_path, index=False)
    
    # Save full processed dataset
    full_path = os.path.join(PROC_DIR, f"{crop_name}_processed.csv")
    df.to_csv(full_path, index=False)
    
    # Identify uninformative zero-variance features
    # (Identical to Paper 1 Section IV: "The columns which contain zero variances were recognized and removed")
    zero_var_cols = [c for c in df.columns if df[c].nunique() <= 1 and c != 'Yield']
    # Also drop Loc_Id and Year from X predictors during model training (preserve pure agroclimatic features)
    drop_from_pruned = zero_var_cols.copy()
    for col in ['Loc_Id', 'Year']:
        if col in df.columns:
            drop_from_pruned.append(col)
            
    pruned_df = df.drop(columns=drop_from_pruned)
    pruned_path = os.path.join(PROC_DIR, f"{crop_name}_pruned.csv")
    pruned_df.to_csv(pruned_path, index=False)
    
    print(f"Zero-variance features pruned: {len(zero_var_cols)} ({zero_var_cols[:5]}...)")
    print(f"Final trainable features in {crop_name}_pruned.csv: {pruned_df.shape[1] - 1} (excluding target 'Yield')")
    return pruned_df

def run_benchmarks_and_tuning(crop_name, pruned_df):
    print(f"\n==========================================")
    print(f"   TRAINING & BENCHMARKING: {crop_name.upper()}")
    print(f"==========================================")
    
    X = pruned_df.drop(columns=['Yield'])
    y = pruned_df['Yield']
    feature_cols = X.columns.tolist()
    
    # 80/20 train/test holdout split with strict random state 42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Train split: {X_train.shape} | Test split: {X_test.shape}")
    
    # 5-fold cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # 1. Define Model Suite
    models = {
        "Multiple Linear Regression": Pipeline([('scaler', StandardScaler()), ('reg', LinearRegression())]),
        "Ridge Regression": Pipeline([('scaler', StandardScaler()), ('reg', Ridge(alpha=100.0, random_state=RANDOM_STATE))]),
        "Lasso Regression": Pipeline([('scaler', StandardScaler()), ('reg', Lasso(alpha=0.1, random_state=RANDOM_STATE, max_iter=2000))]),
        "ElasticNet": Pipeline([('scaler', StandardScaler()), ('reg', ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=2000))]),
        "K-Nearest Neighbors (KNN)": Pipeline([('scaler', StandardScaler()), ('reg', KNeighborsRegressor(n_neighbors=7, weights='distance'))]),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=10, min_samples_leaf=4, random_state=RANDOM_STATE),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=16, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1),
        "Extra Trees Regressor": ExtraTreesRegressor(n_estimators=100, max_depth=16, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting (GBR)": GradientBoostingRegressor(n_estimators=60, learning_rate=0.1, max_depth=4, subsample=0.85, random_state=RANDOM_STATE),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=120, learning_rate=0.08, max_depth=7, random_state=RANDOM_STATE),
        "AdaBoost Regressor": AdaBoostRegressor(n_estimators=50, learning_rate=0.05, random_state=RANDOM_STATE),
        "SVR (RBF Kernel)": Pipeline([('scaler', StandardScaler()), ('reg', SVR(C=50.0, epsilon=0.2))]),
        "XGBoost Regressor (Baseline)": xgb.XGBRegressor(n_estimators=150, learning_rate=0.06, max_depth=6, subsample=0.85, colsample_bytree=0.85, random_state=RANDOM_STATE, n_jobs=-1),
        "LightGBM Regressor": lgb.LGBMRegressor(n_estimators=150, learning_rate=0.06, max_depth=7, subsample=0.85, colsample_bytree=0.85, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    }
    
    results = []
    scoring = {
        'r2': 'r2',
        'neg_rmse': 'neg_root_mean_squared_error',
        'neg_mae': 'neg_mean_absolute_error'
    }
    
    for name, model in models.items():
        print(f"Evaluating {name}...", flush=True)
        t0 = time.time()
        cv_scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        
        cv_r2_mean = float(np.mean(cv_scores['test_r2']))
        cv_r2_std = float(np.std(cv_scores['test_r2']))
        cv_rmse_mean = float(np.mean(-cv_scores['test_neg_rmse']))
        cv_rmse_std = float(np.std(-cv_scores['test_neg_rmse']))
        cv_mae_mean = float(np.mean(-cv_scores['test_neg_mae']))
        
        # Fit on train and predict test
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        test_r2 = float(r2_score(y_test, y_pred))
        test_rmse = float(root_mean_squared_error(y_test, y_pred))
        test_mae = float(mean_absolute_error(y_test, y_pred))
        test_mape = float(mean_absolute_percentage_error(y_test, y_pred)) * 100.0
        
        results.append({
            "Model": name,
            "CV R² (Mean ± Std)": f"{cv_r2_mean:.4f} ± {cv_r2_std:.4f}",
            "CV RMSE (bu/ac)": f"{cv_rmse_mean:.2f} ± {cv_rmse_std:.2f}",
            "CV MAE (bu/ac)": f"{cv_mae_mean:.2f}",
            "Test R²": round(test_r2, 4),
            "Test RMSE (bu/ac)": round(test_rmse, 2),
            "Test MAE (bu/ac)": round(test_mae, 2),
            "Test MAPE (%)": round(test_mape, 2),
            "_sort_r2": test_r2,
            "_sort_rmse": test_rmse
        })
        print(f"   -> Test R²: {test_r2:.4f} | Test RMSE: {test_rmse:.2f} bu/ac ({time.time()-t0:.1f}s)", flush=True)
    
    # 2. Add Ensembles: Voting Regressor & Stacking Regressor (MSER)
    print("Evaluating Voting Regressor...", flush=True)
    top_base_models = [
        ('rf', RandomForestRegressor(n_estimators=80, max_depth=14, random_state=RANDOM_STATE, n_jobs=-1)),
        ('xgb', xgb.XGBRegressor(n_estimators=100, learning_rate=0.07, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1)),
        ('lgb', lgb.LGBMRegressor(n_estimators=100, learning_rate=0.07, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)),
        ('et', ExtraTreesRegressor(n_estimators=80, max_depth=14, random_state=RANDOM_STATE, n_jobs=-1))
    ]
    voting_model = VotingRegressor(estimators=top_base_models)
    v_cv_scores = cross_validate(voting_model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
    voting_model.fit(X_train, y_train)
    v_pred = voting_model.predict(X_test)
    results.append({
        "Model": "Voting Ensemble (RF + XGB + LGB + ET)",
        "CV R² (Mean ± Std)": f"{np.mean(v_cv_scores['test_r2']):.4f} ± {np.std(v_cv_scores['test_r2']):.4f}",
        "CV RMSE (bu/ac)": f"{np.mean(-v_cv_scores['test_neg_rmse']):.2f} ± {np.std(-v_cv_scores['test_neg_rmse']):.2f}",
        "CV MAE (bu/ac)": f"{np.mean(-v_cv_scores['test_neg_mae']):.2f}",
        "Test R²": round(float(r2_score(y_test, v_pred)), 4),
        "Test RMSE (bu/ac)": round(float(root_mean_squared_error(y_test, v_pred)), 2),
        "Test MAE (bu/ac)": round(float(mean_absolute_error(y_test, v_pred)), 2),
        "Test MAPE (%)": round(float(mean_absolute_percentage_error(y_test, v_pred) * 100.0), 2),
        "_sort_r2": float(r2_score(y_test, v_pred)),
        "_sort_rmse": float(root_mean_squared_error(y_test, v_pred))
    })

    print("Evaluating Stacking Regressor (MSER)...", flush=True)
    stacking_model = StackingRegressor(
        estimators=[
            ('rf', RandomForestRegressor(n_estimators=60, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)),
            ('xgb', xgb.XGBRegressor(n_estimators=80, learning_rate=0.08, max_depth=5, random_state=RANDOM_STATE, n_jobs=-1)),
            ('ridge', Ridge(alpha=100.0, random_state=RANDOM_STATE))
        ],
        final_estimator=Ridge(alpha=10.0),
        cv=3
    )
    s_cv_scores = cross_validate(stacking_model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
    stacking_model.fit(X_train, y_train)
    s_pred = stacking_model.predict(X_test)
    results.append({
        "Model": "Stacking Regressor (RF + XGB + Ridge -> Ridge)",
        "CV R² (Mean ± Std)": f"{np.mean(s_cv_scores['test_r2']):.4f} ± {np.std(s_cv_scores['test_r2']):.4f}",
        "CV RMSE (bu/ac)": f"{np.mean(-s_cv_scores['test_neg_rmse']):.2f} ± {np.std(-s_cv_scores['test_neg_rmse']):.2f}",
        "CV MAE (bu/ac)": f"{np.mean(-s_cv_scores['test_neg_mae']):.2f}",
        "Test R²": round(float(r2_score(y_test, s_pred)), 4),
        "Test RMSE (bu/ac)": round(float(root_mean_squared_error(y_test, s_pred)), 2),
        "Test MAE (bu/ac)": round(float(mean_absolute_error(y_test, s_pred)), 2),
        "Test MAPE (%)": round(float(mean_absolute_percentage_error(y_test, s_pred) * 100.0), 2),
        "_sort_r2": float(r2_score(y_test, s_pred)),
        "_sort_rmse": float(root_mean_squared_error(y_test, s_pred))
    })

    # 3. Fine-Tuning the Champion Model (XGBoost)
    print("\n>>> Fine-Tuning XGBoost Hyperparameters via GridSearchCV <<<", flush=True)
    param_grid = {
        'n_estimators': [250, 350],
        'learning_rate': [0.04, 0.07],
        'max_depth': [5, 6],
        'subsample': [0.85],
        'colsample_bytree': [0.85]
    }
    grid_xgb = GridSearchCV(
        estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state=RANDOM_STATE, n_jobs=1),
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error',
        cv=3,
        n_jobs=4,
        verbose=0
    )
    grid_xgb.fit(X_train, y_train)
    best_xgb = grid_xgb.best_estimator_
    print(f"Optimal XGBoost Hyperparameters: {grid_xgb.best_params_}", flush=True)
    
    tuned_cv_scores = cross_validate(best_xgb, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
    best_xgb.fit(X_train, y_train)
    tuned_pred = best_xgb.predict(X_test)
    
    tuned_test_r2 = float(r2_score(y_test, tuned_pred))
    tuned_test_rmse = float(root_mean_squared_error(y_test, tuned_pred))
    tuned_test_mae = float(mean_absolute_error(y_test, tuned_pred))
    tuned_test_mape = float(mean_absolute_percentage_error(y_test, tuned_pred) * 100.0)
    
    results.append({
        "Model": "Tuned XGBoost (Selected Final Champion)",
        "CV R² (Mean ± Std)": f"{np.mean(tuned_cv_scores['test_r2']):.4f} ± {np.std(tuned_cv_scores['test_r2']):.4f}",
        "CV RMSE (bu/ac)": f"{np.mean(-tuned_cv_scores['test_neg_rmse']):.2f} ± {np.std(-tuned_cv_scores['test_neg_rmse']):.2f}",
        "CV MAE (bu/ac)": f"{np.mean(-tuned_cv_scores['test_neg_mae']):.2f}",
        "Test R²": round(tuned_test_r2, 4),
        "Test RMSE (bu/ac)": round(tuned_test_rmse, 2),
        "Test MAE (bu/ac)": round(tuned_test_mae, 2),
        "Test MAPE (%)": round(tuned_test_mape, 2),
        "_sort_r2": tuned_test_r2,
        "_sort_rmse": tuned_test_rmse
    })
    
    df_results = pd.DataFrame(results)
    df_sorted = df_results.sort_values(by='_sort_r2', ascending=False).drop(columns=['_sort_r2', '_sort_rmse'])
    
    csv_path = os.path.join(REPORT_DIR, f"{crop_name}_model_comparison.csv")
    md_path = os.path.join(REPORT_DIR, f"{crop_name}_model_comparison.md")
    df_sorted.to_csv(csv_path, index=False)
    save_markdown_table(df_sorted, md_path)
    
    print(f"\n==================== {crop_name.upper()} BENCHMARK RESULTS ====================")
    print(df_sorted.to_string(index=False))
    
    # 4. Save Final Production Model Artifact
    artifact = {
        "crop": crop_name,
        "model_name": "Tuned XGBoost Regressor",
        "model_version": "1.0.0",
        "best_params": grid_xgb.best_params_,
        "features": feature_cols,
        "n_features": len(feature_cols),
        "model": best_xgb,
        "metrics": {
            "cv_r2": float(np.mean(tuned_cv_scores['test_r2'])),
            "cv_rmse": float(np.mean(-tuned_cv_scores['test_neg_rmse'])),
            "cv_mae": float(np.mean(-tuned_cv_scores['test_neg_mae'])),
            "test_r2": tuned_test_r2,
            "test_rmse": tuned_test_rmse,
            "test_mae": tuned_test_mae,
            "test_mape": tuned_test_mape
        },
        "stats": {
            "yield_min": float(y.min()),
            "yield_mean": float(y.mean()),
            "yield_median": float(y.median()),
            "yield_max": float(y.max()),
            "yield_std": float(y.std())
        },
        "feature_means": X.mean().to_dict(),
        "feature_stds": X.std().to_dict()
    }
    artifact_path = os.path.join(MODEL_DIR, f"{crop_name}_model_final.pkl")
    joblib.dump(artifact, artifact_path)
    print(f"Production model artifact serialized to: {artifact_path}")
    
    # 5. Generate Publication-Quality Visualizations
    # Chart 1: R² Comparison Bar Chart
    plot_df = df_results.sort_values(by='_sort_r2', ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(plot_df['Model'], plot_df['_sort_r2'], color='#2a9d8f')
    bars[-1].set_color('#e76f51') # Highlight champion
    ax.set_xlim(0, 1.0)
    ax.set_xlabel('Test R² Score', fontsize=12, fontweight='bold')
    ax.set_title(f'{crop_name.capitalize()} Yield Prediction: 17 Regression Models Ranked by R²', fontsize=14, fontweight='bold')
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}', va='center', fontsize=9.5)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"{crop_name}_model_r2_comparison.png"), dpi=300)
    plt.close()
    
    # Chart 2: Actual vs Predicted Residual Diagnostics
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_test, tuned_pred, alpha=0.35, color='#264653', edgecolors='none', s=35)
    min_val = min(y_test.min(), tuned_pred.min())
    max_val = max(y_test.max(), tuned_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 Perfect Fit')
    axes[0].set_title(f'{crop_name.capitalize()}: Actual vs Predicted Yield (Test Set)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Actual Yield (bu/acre)', fontsize=11)
    axes[0].set_ylabel('Predicted Yield (bu/acre)', fontsize=11)
    axes[0].legend()
    
    residuals = y_test - tuned_pred
    sns.histplot(residuals, kde=True, ax=axes[1], color='#e76f51', bins=30)
    axes[1].axvline(0, color='black', linestyle='--')
    axes[1].set_title(f'{crop_name.capitalize()}: Prediction Residual Distribution', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Residual (Actual - Predicted, bu/acre)', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"{crop_name}_actual_vs_predicted.png"), dpi=300)
    plt.close()
    
    return artifact

if __name__ == '__main__':
    t_start = time.time()
    print("=" * 60)
    print("  KRISHIYIELD MASTER PIPELINE: PREPROCESSING & TRAINING")
    print("=" * 60)
    
    # 1. Preprocess both crops
    corn_pruned = preprocess_dataset("corn_data_csv.csv", "corn")
    soybean_pruned = preprocess_dataset("Soybean_data_csv.csv", "soybean")
    
    # 2. Benchmark all regression algorithms & fine-tune champion
    corn_artifact = run_benchmarks_and_tuning("corn", corn_pruned)
    soy_artifact = run_benchmarks_and_tuning("soybean", soybean_pruned)
    
    total_time = (time.time() - t_start) / 60.0
    print(f"\n============================================================")
    print(f" PIPELINE COMPLETE IN {total_time:.1f} MINUTES!")
    print(f" Corn Model R²: {corn_artifact['metrics']['test_r2']:.4f} | RMSE: {corn_artifact['metrics']['test_rmse']:.2f} bu/ac")
    print(f" Soybean Model R²: {soy_artifact['metrics']['test_r2']:.4f} | RMSE: {soy_artifact['metrics']['test_rmse']:.2f} bu/ac")
    print(f"============================================================")
