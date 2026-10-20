import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Setup paths
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(PROJ_DIR, "data", "raw")
PROC_DIR = os.path.join(PROJ_DIR, "data", "processed")
FIG_DIR = os.path.join(PROJ_DIR, "reports", "figures")
os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = 11

print("--- Loading Raw Datasets ---")
df_corn_raw = pd.read_csv(os.path.join(RAW_DIR, "corn_data_csv.csv"))
df_soy_raw = pd.read_csv(os.path.join(RAW_DIR, "Soybean_data_csv.csv"))

# Fix anomalous column '*' in Corn
if '*' in df_corn_raw.columns:
    df_corn_raw = df_corn_raw.rename(columns={'*': 'W34'})
    print("Fixed anomalous column '*' -> 'W34' in Corn dataset.")

# Check zero-dominant features (e.g. summer snow water equivalent W316-W345)
def identify_zero_dominant(df, threshold=0.98):
    zero_dominant = []
    for col in df.columns:
        if col not in ['Loc_Id', 'Year', 'Yield']:
            zero_ratio = (df[col] == 0).mean()
            if zero_ratio >= threshold:
                zero_dominant.append((col, zero_ratio))
    return zero_dominant

corn_zeros = identify_zero_dominant(df_corn_raw)
soy_zeros = identify_zero_dominant(df_soy_raw)
print(f"Corn zero-dominant features (>=98% zeros): {len(corn_zeros)} columns")
print(f"Soybean zero-dominant features (>=98% zeros): {len(soy_zeros)} columns")

corn_drop_zero = [c[0] for c in corn_zeros]
soy_drop_zero = [c[0] for c in soy_zeros]

# Create Clean Processed Datasets
# 1. Full cleaned datasets (keeping all informative features, fixing column names)
corn_clean = df_corn_raw.drop(columns=['Loc_Id', 'Year'])
soy_clean = df_soy_raw.drop(columns=['Loc_Id', 'Year'])

# 2. Pruned datasets without zero-dominant features (used for high-performance boosted tree benchmarks)
corn_pruned = corn_clean.drop(columns=corn_drop_zero)
soy_pruned = soy_clean.drop(columns=soy_drop_zero)

print(f"Corn Clean shape: {corn_clean.shape}, Corn Pruned shape: {corn_pruned.shape}")
print(f"Soy Clean shape: {soy_clean.shape}, Soy Pruned shape: {soy_pruned.shape}")

# Save Processed Datasets
corn_clean.to_csv(os.path.join(PROC_DIR, "corn_processed.csv"), index=False)
corn_pruned.to_csv(os.path.join(PROC_DIR, "corn_pruned.csv"), index=False)
soy_clean.to_csv(os.path.join(PROC_DIR, "soybean_processed.csv"), index=False)
soy_pruned.to_csv(os.path.join(PROC_DIR, "soybean_pruned.csv"), index=False)

# Also preserve metadata (Loc_Id, Year, Yield) for group/time cross validation
df_corn_raw[['Loc_Id', 'Year', 'Yield']].to_csv(os.path.join(PROC_DIR, "corn_meta.csv"), index=False)
df_soy_raw[['Loc_Id', 'Year', 'Yield']].to_csv(os.path.join(PROC_DIR, "soybean_meta.csv"), index=False)
print("Saved processed CSVs to data/processed/")

# ----------------- GENERATE EDA PLOTS -----------------
print("--- Generating EDA Visualizations ---")

# 1. Target Yield Distributions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df_corn_raw['Yield'], kde=True, ax=axes[0], color='#2d7a46', bins=35)
axes[0].set_title('Corn Yield Distribution (1980–2018)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Yield (bushels/acre)', fontsize=11)
axes[0].set_ylabel('County-Year Frequency', fontsize=11)
axes[0].axvline(df_corn_raw['Yield'].mean(), color='#e63946', linestyle='--', label=f'Mean: {df_corn_raw["Yield"].mean():.1f} bu/ac')
axes[0].axvline(df_corn_raw['Yield'].median(), color='#457b9d', linestyle=':', label=f'Median: {df_corn_raw["Yield"].median():.1f} bu/ac')
axes[0].legend()

sns.histplot(df_soy_raw['Yield'], kde=True, ax=axes[1], color='#e76f51', bins=35)
axes[1].set_title('Soybean Yield Distribution (1980–2018)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Yield (bushels/acre)', fontsize=11)
axes[1].set_ylabel('County-Year Frequency', fontsize=11)
axes[1].axvline(df_soy_raw['Yield'].mean(), color='#e63946', linestyle='--', label=f'Mean: {df_soy_raw["Yield"].mean():.1f} bu/ac')
axes[1].axvline(df_soy_raw['Yield'].median(), color='#457b9d', linestyle=':', label=f'Median: {df_soy_raw["Yield"].median():.1f} bu/ac')
axes[1].legend()

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "yield_distributions.png"), dpi=300)
plt.close()
print("Saved yield_distributions.png")

# 2. Critical Mid-Season Maximum Temperature vs Yield (Weeks 28-32: July Flowering)
# W429 is Week 29 Max Temp (~mid-July)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.regplot(data=df_corn_raw, x='W429', y='Yield', ax=axes[0],
            scatter_kws={'alpha': 0.25, 'color': '#2a9d8f'}, line_kws={'color': '#d62828', 'linewidth': 2})
axes[0].set_title('Corn: Mid-Season Max Temp (Week 29, July) vs Yield', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Max Temperature Week 29 (°C)', fontsize=11)
axes[0].set_ylabel('Yield (bu/acre)', fontsize=11)

sns.regplot(data=df_soy_raw, x='W430', y='Yield', ax=axes[1],
            scatter_kws={'alpha': 0.25, 'color': '#f4a261'}, line_kws={'color': '#d62828', 'linewidth': 2})
axes[1].set_title('Soybean: Mid-Season Max Temp (Week 30, Late July) vs Yield', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Max Temperature Week 30 (°C)', fontsize=11)
axes[1].set_ylabel('Yield (bu/acre)', fontsize=11)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "temperature_vs_yield.png"), dpi=300)
plt.close()
print("Saved temperature_vs_yield.png")

# 3. Topsoil Organic Matter & Soil Productivity vs Yield
# S71 is 0-5cm organic matter, S_surface2 is productivity index
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.regplot(data=df_corn_raw, x='S_surface2', y='Yield', ax=axes[0],
            scatter_kws={'alpha': 0.25, 'color': '#1d3557'}, line_kws={'color': '#2a9d8f', 'linewidth': 2})
axes[0].set_title('Corn: Soil Productivity Index (S_surface2) vs Yield', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Soil Productivity Index', fontsize=11)
axes[0].set_ylabel('Yield (bu/acre)', fontsize=11)

sns.regplot(data=df_soy_raw, x='S_surface2', y='Yield', ax=axes[1],
            scatter_kws={'alpha': 0.25, 'color': '#606c38'}, line_kws={'color': '#283618', 'linewidth': 2})
axes[1].set_title('Soybean: Soil Productivity Index (S_surface2) vs Yield', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Soil Productivity Index', fontsize=11)
axes[1].set_ylabel('Yield (bu/acre)', fontsize=11)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "soil_productivity_vs_yield.png"), dpi=300)
plt.close()
print("Saved soil_productivity_vs_yield.png")

# 4. Top Feature Correlation Heatmaps
corn_corr = corn_clean.corr()['Yield'].drop('Yield')
top_corn_corr = corn_corr.abs().sort_values(ascending=False).head(20).index
corn_corr_subset = corn_clean[list(top_corn_corr) + ['Yield']].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corn_corr_subset, annot=False, cmap='coolwarm', center=0, ax=ax, cbar_kws={'label': 'Pearson Correlation'})
ax.set_title('Top 20 Features Correlated with Corn Yield', fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "corn_top20_correlation_heatmap.png"), dpi=300)
plt.close()
print("Saved corn_top20_correlation_heatmap.png")

soy_corr = soy_clean.corr()['Yield'].drop('Yield')
top_soy_corr = soy_corr.abs().sort_values(ascending=False).head(20).index
soy_corr_subset = soy_clean[list(top_soy_corr) + ['Yield']].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(soy_corr_subset, annot=False, cmap='coolwarm', center=0, ax=ax, cbar_kws={'label': 'Pearson Correlation'})
ax.set_title('Top 20 Features Correlated with Soybean Yield', fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "soybean_top20_correlation_heatmap.png"), dpi=300)
plt.close()
print("Saved soybean_top20_correlation_heatmap.png")

# 5. Temporal Yield Trend (1980 - 2018)
fig, ax = plt.subplots(figsize=(12, 5))
corn_annual = df_corn_raw.groupby('Year')['Yield'].agg(['mean', 'std']).reset_index()
soy_annual = df_soy_raw.groupby('Year')['Yield'].agg(['mean', 'std']).reset_index()

ax.plot(corn_annual['Year'], corn_annual['mean'], label='Corn Mean Yield (bu/acre)', color='#2d7a46', linewidth=2.5, marker='o')
ax.fill_between(corn_annual['Year'], corn_annual['mean'] - corn_annual['std'], corn_annual['mean'] + corn_annual['std'], color='#2d7a46', alpha=0.15)

ax.plot(soy_annual['Year'], soy_annual['mean'], label='Soybean Mean Yield (bu/acre)', color='#e76f51', linewidth=2.5, marker='s')
ax.fill_between(soy_annual['Year'], soy_annual['mean'] - soy_annual['std'], soy_annual['mean'] + soy_annual['std'], color='#e76f51', alpha=0.15)

# Highlight historic 2012 drought
ax.axvspan(2011.5, 2012.5, color='#d62828', alpha=0.2, label='2012 US Severe Drought')
ax.set_title('Historical Crop Yield Trajectory (1980–2018)', fontsize=14, fontweight='bold')
ax.set_xlabel('Harvest Year', fontsize=12)
ax.set_ylabel('Yield (bushels/acre)', fontsize=12)
ax.legend(loc='upper left', frameon=True)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "historical_yield_trends.png"), dpi=300)
plt.close()
print("Saved historical_yield_trends.png")

print("--- All Data Preprocessing and EDA Visualizations Completed! ---")
