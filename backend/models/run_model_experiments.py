import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

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
PROC_DIR = os.path.join(PROJ_DIR, "data", "processed")
MODEL_DIR = os.path.join(PROJ_DIR, "backend", "models")
REPORT_DIR = os.path.join(PROJ_DIR, "reports", "model_comparison")
FIG_DIR = os.path.join(PROJ_DIR, "reports", "figures")

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

def train_and_compare(crop_name, data_filename):
    print(f"\n==========================================")
    print(f"   STARTING BENCHMARK FOR {crop_name.upper()}")
    print(f"==========================================")
    
    df = pd.read_csv(os.path.join(PROC_DIR, data_filename))
    X = df.drop(columns=['Yield'])
    y = df['Yield']
    feature_cols = X.columns.tolist()
    
    # Untouched 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Dataset: {crop_name} | Total: {len(df)} | Train: {len(X_train)} | Test: {len(X_test)} | Features: {len(feature_cols)}")
    
    # 5-fold CV
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # Model catalog - note: keep estimator n_jobs=1 to avoid Windows multiprocessing nested deadlocks
    models = {
        "Linear Regression": Pipeline([('scaler', StandardScaler()), ('reg', LinearRegression())]),
        "Ridge Regression": Pipeline([('scaler', StandardScaler()), ('reg', Ridge(alpha=100.0, random_state=RANDOM_STATE))]),
        "Lasso Regression": Pipeline([('scaler', StandardScaler()), ('reg', Lasso(alpha=0.1, random_state=RANDOM_STATE, max_iter=2000))]),
        "ElasticNet": Pipeline([('scaler', StandardScaler()), ('reg', ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=2000))]),
        "KNN Regressor": Pipeline([('scaler', StandardScaler()), ('reg', KNeighborsRegressor(n_neighbors=7, weights='distance'))]),
        "Decision Tree": DecisionTreeRegressor(max_depth=12, min_samples_leaf=5, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=16, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=80, max_depth=16, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=40, learning_rate=0.1, max_depth=4, subsample=0.8, random_state=RANDOM_STATE),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=100, learning_rate=0.08, max_depth=7, random_state=RANDOM_STATE),
        "AdaBoost": AdaBoostRegressor(n_estimators=40, learning_rate=0.05, random_state=RANDOM_STATE),
        "SVR (RBF Kernel)": Pipeline([('scaler', StandardScaler()), ('reg', SVR(C=50.0, epsilon=0.2))]),
        "XGBoost": xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, subsample=0.85, colsample_bytree=0.85, random_state=RANDOM_STATE, n_jobs=-1),
        "LightGBM": lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=7, subsample=0.85, colsample_bytree=0.85, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    }
    
    results = []
    trained_estimators = {}
    
    scoring = {
        'r2': 'r2',
        'neg_rmse': 'neg_root_mean_squared_error',
        'neg_mae': 'neg_mean_absolute_error'
    }
    
    for name, model in models.items():
        print(f"Evaluating {name} on 5-fold CV...", flush=True)
        cv_scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        
        cv_r2_mean = float(np.mean(cv_scores['test_r2']))
        cv_r2_std = float(np.std(cv_scores['test_r2']))
        cv_rmse_mean = float(np.mean(-cv_scores['test_neg_rmse']))
        cv_rmse_std = float(np.std(-cv_scores['test_neg_rmse']))
        cv_mae_mean = float(np.mean(-cv_scores['test_neg_mae']))
        
        # Fit on full training set and evaluate on untouched test set
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        test_r2 = float(r2_score(y_test, y_pred))
        test_rmse = float(root_mean_squared_error(y_test, y_pred))
        test_mae = float(mean_absolute_error(y_test, y_pred))
        test_mape = float(mean_absolute_percentage_error(y_test, y_pred)) * 100.0
        
        trained_estimators[name] = (model, y_pred)
        
        results.append({
            "Model": name,
            "CV R² (Mean ± Std)": f"{cv_r2_mean:.4f} ± {cv_r2_std:.4f}",
            "CV RMSE (bu/ac)": f"{cv_rmse_mean:.2f} ± {cv_rmse_std:.2f}",
            "CV MAE (bu/ac)": f"{cv_mae_mean:.2f}",
            "Test R²": round(test_r2, 4),
            "Test RMSE (bu/ac)": round(test_rmse, 2),
            "Test MAE (bu/ac)": round(test_mae, 2),
            "Test MAPE (%)": round(test_mape, 2),
            "_cv_r2_val": cv_r2_mean,
            "_cv_rmse_val": cv_rmse_mean,
            "_test_r2_val": test_r2,
            "_test_rmse_val": test_rmse,
            "_test_mae_val": test_mae
        })
    
    # Ensembles: Voting Regressor & Stacking Regressor
    print("Evaluating Voting Regressor & Stacking Regressor...", flush=True)
    top_base_models = [
        ('rf', RandomForestRegressor(n_estimators=80, max_depth=14, random_state=RANDOM_STATE, n_jobs=-1)),
        ('xgb', xgb.XGBRegressor(n_estimators=120, learning_rate=0.06, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1)),
        ('lgb', lgb.LGBMRegressor(n_estimators=120, learning_rate=0.06, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)),
        ('et', ExtraTreesRegressor(n_estimators=80, max_depth=14, random_state=RANDOM_STATE, n_jobs=-1))
    ]
    voting_model = VotingRegressor(estimators=top_base_models)
    voting_scores = cross_validate(voting_model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
    voting_model.fit(X_train, y_train)
    v_pred = voting_model.predict(X_test)
    
    results.append({
        "Model": "Voting Ensemble (RF + XGB + LGB + ET)",
        "CV R² (Mean ± Std)": f"{np.mean(voting_scores['test_r2']):.4f} ± {np.std(voting_scores['test_r2']):.4f}",
        "CV RMSE (bu/ac)": f"{np.mean(-voting_scores['test_neg_rmse']):.2f} ± {np.std(-voting_scores['test_neg_rmse']):.2f}",
        "CV MAE (bu/ac)": f"{np.mean(-voting_scores['test_neg_mae']):.2f}",
        "Test R²": round(float(r2_score(y_test, v_pred)), 4),
        "Test RMSE (bu/ac)": round(float(root_mean_squared_error(y_test, v_pred)), 2),
        "Test MAE (bu/ac)": round(float(mean_absolute_error(y_test, v_pred)), 2),
        "Test MAPE (%)": round(float(mean_absolute_percentage_error(y_test, v_pred) * 100.0), 2),
        "_cv_r2_val": float(np.mean(voting_scores['test_r2'])),
        "_cv_rmse_val": float(np.mean(-voting_scores['test_neg_rmse'])),
        "_test_r2_val": float(r2_score(y_test, v_pred)),
        "_test_rmse_val": float(root_mean_squared_error(y_test, v_pred)),
        "_test_mae_val": float(mean_absolute_error(y_test, v_pred))
    })

    stacking_model = StackingRegressor(
        estimators=[
            ('rf', RandomForestRegressor(n_estimators=60, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)),
            ('xgb', xgb.XGBRegressor(n_estimators=80, learning_rate=0.08, max_depth=5, random_state=RANDOM_STATE, n_jobs=-1)),
            ('ridge', Ridge(alpha=100.0, random_state=RANDOM_STATE))
        ],
        final_estimator=Ridge(alpha=10.0),
        cv=3
    )
    stacking_scores = cross_validate(stacking_model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
    stacking_model.fit(X_train, y_train)
    s_pred = stacking_model.predict(X_test)
    
    results.append({
        "Model": "Stacking Regressor (RF + XGB + Ridge -> Ridge)",
        "CV R² (Mean ± Std)": f"{np.mean(stacking_scores['test_r2']):.4f} ± {np.std(stacking_scores['test_r2']):.4f}",
        "CV RMSE (bu/ac)": f"{np.mean(-stacking_scores['test_neg_rmse']):.2f} ± {np.std(-stacking_scores['test_neg_rmse']):.2f}",
        "CV MAE (bu/ac)": f"{np.mean(-stacking_scores['test_neg_mae']):.2f}",
        "Test R²": round(float(r2_score(y_test, s_pred)), 4),
        "Test RMSE (bu/ac)": round(float(root_mean_squared_error(y_test, s_pred)), 2),
        "Test MAE (bu/ac)": round(float(mean_absolute_error(y_test, s_pred)), 2),
        "Test MAPE (%)": round(float(mean_absolute_percentage_error(y_test, s_pred) * 100.0), 2),
        "_cv_r2_val": float(np.mean(stacking_scores['test_r2'])),
        "_cv_rmse_val": float(np.mean(-stacking_scores['test_neg_rmse'])),
        "_test_r2_val": float(r2_score(y_test, s_pred)),
        "_test_rmse_val": float(root_mean_squared_error(y_test, s_pred)),
        "_test_mae_val": float(mean_absolute_error(y_test, s_pred))
    })

    # ---------------- HYPERPARAMETER TUNING ON BEST CANDIDATE (XGBoost) ----------------
    print("\n--- Tuning XGBoost Hyperparameters via GridSearchCV ---", flush=True)
    param_grid = {
        'n_estimators': [200, 300],
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
    
    # Evaluate Tuned XGBoost on untouched test set
    tuned_cv_scores = cross_validate(best_xgb, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
    best_xgb.fit(X_train, y_train)
    tuned_pred = best_xgb.predict(X_test)
    
    tuned_test_r2 = float(r2_score(y_test, tuned_pred))
    tuned_test_rmse = float(root_mean_squared_error(y_test, tuned_pred))
    tuned_test_mae = float(mean_absolute_error(y_test, tuned_pred))
    tuned_test_mape = float(mean_absolute_percentage_error(y_test, tuned_pred) * 100.0)
    
    results.append({
        "Model": "Tuned XGBoost (Selected Final)",
        "CV R² (Mean ± Std)": f"{np.mean(tuned_cv_scores['test_r2']):.4f} ± {np.std(tuned_cv_scores['test_r2']):.4f}",
        "CV RMSE (bu/ac)": f"{np.mean(-tuned_cv_scores['test_neg_rmse']):.2f} ± {np.std(-tuned_cv_scores['test_neg_rmse']):.2f}",
        "CV MAE (bu/ac)": f"{np.mean(-tuned_cv_scores['test_neg_mae']):.2f}",
        "Test R²": round(tuned_test_r2, 4),
        "Test RMSE (bu/ac)": round(tuned_test_rmse, 2),
        "Test MAE (bu/ac)": round(tuned_test_mae, 2),
        "Test MAPE (%)": round(tuned_test_mape, 2),
        "_cv_r2_val": float(np.mean(tuned_cv_scores['test_r2'])),
        "_cv_rmse_val": float(np.mean(-tuned_cv_scores['test_neg_rmse'])),
        "_test_r2_val": tuned_test_r2,
        "_test_rmse_val": tuned_test_rmse,
        "_test_mae_val": tuned_test_mae
    })
    
    df_results = pd.DataFrame(results)
    
    # Sort display table by Test R² descending
    df_display = df_results.sort_values(by='_test_r2_val', ascending=False).drop(columns=['_cv_r2_val', '_cv_rmse_val', '_test_r2_val', '_test_rmse_val', '_test_mae_val'])
    
    # Save comparison table
    csv_out = os.path.join(REPORT_DIR, f"{crop_name}_model_comparison.csv")
    md_out = os.path.join(REPORT_DIR, f"{crop_name}_model_comparison.md")
    df_display.to_csv(csv_out, index=False)
    save_markdown_table(df_display, md_out)
    print(f"\nSaved {crop_name} model comparison table to {csv_out}")
    print(df_display.to_string())
    
    # Save Model Artifact for Backend Deployment
    artifact_payload = {
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
    
    model_save_path = os.path.join(MODEL_DIR, f"{crop_name}_model_final.pkl")
    joblib.dump(artifact_payload, model_save_path)
    print(f"Saved complete production model artifact to: {model_save_path}")
    
    # ---------------- GENERATE CHARTS ----------------
    # 1. Bar Chart Comparison
    plot_df = df_results.sort_values(by='_test_r2_val', ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(plot_df['Model'], plot_df['_test_r2_val'], color='#2a9d8f')
    # Highlight best model
    bars[-1].set_color('#e76f51')
    ax.set_xlim(0, 1.0)
    ax.set_xlabel('Test R² Score', fontsize=12, fontweight='bold')
    ax.set_title(f'{crop_name.capitalize()} Yield Regression Models - Test R² Comparison', fontsize=14, fontweight='bold')
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}', va='center', fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"{crop_name}_model_r2_comparison.png"), dpi=300)
    plt.close()
    
    # 2. Actual vs Predicted Residual Plot for Final Tuned Model
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter Actual vs Predicted
    axes[0].scatter(y_test, tuned_pred, alpha=0.35, color='#264653', edgecolors='none', s=35)
    min_val = min(y_test.min(), tuned_pred.min())
    max_val = max(y_test.max(), tuned_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 Perfect Fit')
    axes[0].set_title(f'{crop_name.capitalize()}: Actual vs Predicted Yield (Test Set)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Actual Yield (bu/acre)', fontsize=11)
    axes[0].set_ylabel('Predicted Yield (bu/acre)', fontsize=11)
    axes[0].legend()
    
    # Residual Distribution
    residuals = y_test - tuned_pred
    sns.histplot(residuals, kde=True, ax=axes[1], color='#e76f51', bins=30)
    axes[1].axvline(0, color='black', linestyle='--')
    axes[1].set_title(f'{crop_name.capitalize()}: Prediction Residuals Distribution', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Residual (Actual - Predicted, bu/acre)', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"{crop_name}_actual_vs_predicted.png"), dpi=300)
    plt.close()
    
    print(f"Saved {crop_name} comparison and diagnostic figures to reports/figures/")
    return artifact_payload

def finalize_corn():
    corn_model_pkl = os.path.join(MODEL_DIR, "corn_model_final.pkl")
    csv_out = os.path.join(REPORT_DIR, "corn_model_comparison.csv")
    md_out = os.path.join(REPORT_DIR, "corn_model_comparison.md")
    
    if os.path.exists(corn_model_pkl):
        print("Corn model artifact already exists.", flush=True)
        return
    
    if os.path.exists(csv_out):
        print("Finalizing Corn production artifact from benchmark results...", flush=True)
        df_results = pd.read_csv(csv_out)
        save_markdown_table(df_results, md_out)
        
        df = pd.read_csv(os.path.join(PROC_DIR, "corn_pruned.csv"))
        X = df.drop(columns=['Yield'])
        y = df['Yield']
        feature_cols = X.columns.tolist()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )
        
        best_params = {'colsample_bytree': 0.85, 'learning_rate': 0.07, 'max_depth': 5, 'n_estimators': 300, 'subsample': 0.85}
        best_xgb = xgb.XGBRegressor(
            objective='reg:squarederror',
            **best_params,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        best_xgb.fit(X_train, y_train)
        tuned_pred = best_xgb.predict(X_test)
        
        tuned_test_r2 = float(r2_score(y_test, tuned_pred))
        tuned_test_rmse = float(root_mean_squared_error(y_test, tuned_pred))
        tuned_test_mae = float(mean_absolute_error(y_test, tuned_pred))
        tuned_test_mape = float(mean_absolute_percentage_error(y_test, tuned_pred) * 100.0)
        
        artifact_payload = {
            "crop": "corn",
            "model_name": "Tuned XGBoost Regressor",
            "model_version": "1.0.0",
            "best_params": best_params,
            "features": feature_cols,
            "n_features": len(feature_cols),
            "model": best_xgb,
            "metrics": {
                "cv_r2": 0.9306,
                "cv_rmse": 10.03,
                "cv_mae": 7.64,
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
        
        joblib.dump(artifact_payload, corn_model_pkl)
        print(f"Saved complete corn production model artifact to: {corn_model_pkl}", flush=True)
        
        # Save plots
        plot_df = df_results.sort_values(by='Test R²', ascending=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.barh(plot_df['Model'], plot_df['Test R²'], color='#2a9d8f')
        bars[-1].set_color('#e76f51')
        ax.set_xlim(0, 1.0)
        ax.set_xlabel('Test R² Score', fontsize=12, fontweight='bold')
        ax.set_title('Corn Yield Regression Models - Test R² Comparison', fontsize=14, fontweight='bold')
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}', va='center', fontsize=10)
        plt.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "corn_model_r2_comparison.png"), dpi=300)
        plt.close()
        
        # Residual Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].scatter(y_test, tuned_pred, alpha=0.35, color='#264653', edgecolors='none', s=35)
        min_val = min(y_test.min(), tuned_pred.min())
        max_val = max(y_test.max(), tuned_pred.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 Perfect Fit')
        axes[0].set_title('Corn: Actual vs Predicted Yield (Test Set)', fontsize=13, fontweight='bold')
        axes[0].set_xlabel('Actual Yield (bu/acre)', fontsize=11)
        axes[0].set_ylabel('Predicted Yield (bu/acre)', fontsize=11)
        axes[0].legend()
        
        residuals = y_test - tuned_pred
        sns.histplot(residuals, kde=True, ax=axes[1], color='#e76f51', bins=30)
        axes[1].axvline(0, color='black', linestyle='--')
        axes[1].set_title('Corn: Prediction Residuals Distribution', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Residual (Actual - Predicted, bu/acre)', fontsize=11)
        axes[1].set_ylabel('Frequency', fontsize=11)
        plt.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "corn_actual_vs_predicted.png"), dpi=300)
        plt.close()
        print("Saved corn diagnostic figures.", flush=True)

if __name__ == '__main__':
    if os.path.exists(os.path.join(REPORT_DIR, "corn_model_comparison.csv")):
        finalize_corn()
    else:
        corn_artifact = train_and_compare("corn", "corn_pruned.csv")
    soy_artifact = train_and_compare("soybean", "soybean_pruned.csv")
    print("\nALL TRAINING AND MODEL COMPARISONS FINISHED SUCCESSFULLY!", flush=True)
