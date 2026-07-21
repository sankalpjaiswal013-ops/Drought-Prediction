import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

print("=" * 60)
print("  Future Drought Forecast: Predicting SPI until 2030")
print("=" * 60)

# ---------------------------------------------------------
# 1. Load all processed data and compute yearly mean SPI
# ---------------------------------------------------------
df = pd.read_csv("data/processed_features.csv", parse_dates=['time'])
df['year'] = df['time'].dt.year

# Define yearly aggregation metrics dynamically based on available features
agg_dict = {
    'SPI_mean': ('SPI', 'mean'),
    'Rainfall_mean': ('Rainfall', 'mean'),
    'MaxTemp_mean': ('Max_Temp', 'mean'),
    'SoilMoisture_mean': ('Soil_Moisture', 'mean'),
    'SPI_min': ('SPI', 'min'),
    'SPI_max': ('SPI', 'max'),
    'SPI_std': ('SPI', 'std'),
    'Rainfall_total': ('Rainfall', 'sum'),
}
if 'RH' in df.columns:
    agg_dict['RH_mean'] = ('RH', 'mean')

yearly = df.groupby('year').agg(**agg_dict).reset_index()

print(f"\nYearly aggregated data: {len(yearly)} years (1981-2023)")
print(yearly[['year', 'SPI_mean']].tail(10).to_string(index=False))

# ---------------------------------------------------------
# 2. Build autoregressive features (lag-based)
#    Use past N years of SPI + climate stats to predict next year
# ---------------------------------------------------------
LAG_YEARS = 3  # Use past 3 years to predict next year

feature_cols = ['SPI_mean', 'Rainfall_mean', 'MaxTemp_mean', 'SoilMoisture_mean',
                'SPI_min', 'SPI_max', 'SPI_std', 'Rainfall_total']
if 'RH' in df.columns:
    feature_cols.append('RH_mean')

X_list, y_list, year_list = [], [], []

for i in range(LAG_YEARS, len(yearly)):
    row_features = []
    for lag in range(1, LAG_YEARS + 1):
        for col in feature_cols:
            row_features.append(yearly[col].iloc[i - lag])
    X_list.append(row_features)
    y_list.append(yearly['SPI_mean'].iloc[i])
    year_list.append(yearly['year'].iloc[i])

X = np.array(X_list)
y = np.array(y_list)
years_arr = np.array(year_list)

# Create feature names for interpretability
feature_names = []
for lag in range(1, LAG_YEARS + 1):
    for col in feature_cols:
        feature_names.append(f"{col}_lag{lag}")

print(f"\nAutoregressive dataset: {len(X)} samples, {X.shape[1]} features per sample")
print(f"Years covered: {years_arr[0]} to {years_arr[-1]}")

# ---------------------------------------------------------
# 3. Train on ALL data (using PCA to prevent overfitting)
# ---------------------------------------------------------
from sklearn.decomposition import PCA

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_sc = scaler_X.fit_transform(X)
y_sc = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# Fit PCA to reduce features to 2 components
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_sc)
explained_var = np.sum(pca.explained_variance_ratio_) * 100
print(f"\nApplying PCA dimensionality reduction: {X_sc.shape[1]} dims → 2 dims (explains {explained_var:.1f}% variance)")

# Train multiple models and ensemble them for robust forecasting
models = {}

print("\nTraining forecast models on ALL historical data (in PCA space)...")

# Random Forest
rf = RandomForestRegressor(n_estimators=200, max_depth=4, random_state=42)
rf.fit(X_pca, y_sc)
models['Random Forest'] = rf
print("  ✓ Random Forest trained")

# Gradient Boosting
gb = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
gb.fit(X_pca, y_sc)
models['Gradient Boosting'] = gb
print("  ✓ Gradient Boosting trained")

# MLP
mlp = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42, 
                   early_stopping=True, validation_fraction=0.15)
mlp.fit(X_pca, y_sc)
models['MLP'] = mlp
print("  ✓ MLP trained")

# SVR
svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svr.fit(X_pca, y_sc)
models['SVR'] = svr
print("  ✓ SVR trained")

# ---------------------------------------------------------
# 4. Validate on training data (sanity check)
# ---------------------------------------------------------
print("\n--- Training Fit (Sanity Check) ---")
for name, model in models.items():
    pred_sc = model.predict(X_pca)
    pred = scaler_y.inverse_transform(pred_sc.reshape(-1, 1)).ravel()
    rmse = np.sqrt(mean_squared_error(y, pred))
    r2 = r2_score(y, pred)
    print(f"  {name}: R²={r2:.3f}, RMSE={rmse:.4f}")

# ---------------------------------------------------------
# 5. Recursive Forecasting: 2024 → 2030
# ---------------------------------------------------------
print("\n--- Generating Future Forecast (2024-2030) ---")

# Build a rolling buffer of the most recent LAG_YEARS of data
recent_data = yearly[feature_cols].tail(LAG_YEARS).values.tolist()

forecast_years = list(range(2024, 2031))
forecast_results = {name: [] for name in models}
ensemble_forecast = []

for target_year in forecast_years:
    # Build feature vector from recent_data buffer
    row_features = []
    for lag in range(1, LAG_YEARS + 1):
        idx = len(recent_data) - lag
        for val in recent_data[idx]:
            row_features.append(val)
    
    row_features = np.array(row_features).reshape(1, -1)
    row_sc = scaler_X.transform(row_features)
    row_pca = pca.transform(row_sc)
    
    # Predict with each model
    year_predictions = []
    for name, model in models.items():
        pred_sc = model.predict(row_pca)
        pred = scaler_y.inverse_transform(pred_sc.reshape(-1, 1)).ravel()[0]
        forecast_results[name].append(pred)
        year_predictions.append(pred)
    
    # Ensemble = mean of all models
    ensemble_val = np.mean(year_predictions)
    ensemble_forecast.append(ensemble_val)
    
    # For the next iteration, we need to append a "synthetic" row to recent_data
    # We use the ensemble SPI prediction and extrapolate climate variables from recent trends
    last_row = recent_data[-1]
    # Slightly decay climate features toward their historical mean (conservative assumption)
    hist_means = yearly[feature_cols].mean().values
    alpha = 0.7  # weight toward recent data vs historical mean
    synthetic_row = [ensemble_val]  # SPI_mean = our prediction
    for j in range(1, len(feature_cols)):
        synthetic_row.append(alpha * last_row[j] + (1 - alpha) * hist_means[j])
    
    recent_data.append(synthetic_row)
    
    print(f"  {target_year}: Ensemble SPI = {ensemble_val:+.3f}", end="")
    # Interpret
    if ensemble_val > 0:
        print(" → Normal/Wet")
    elif ensemble_val > -1.0:
        print(" → Mildly Dry")
    elif ensemble_val > -1.5:
        print(" → Moderate Drought ⚠️")
    else:
        print(" → Severe Drought 🚨")

# ---------------------------------------------------------
# 6. Save forecast results
# ---------------------------------------------------------
os.makedirs("outputs/results", exist_ok=True)

forecast_df = pd.DataFrame({
    'Year': forecast_years,
    'Ensemble_SPI': ensemble_forecast,
})
for name in models:
    forecast_df[name] = forecast_results[name]

forecast_df.to_csv("outputs/results/forecast_2024_2030.csv", index=False)
print(f"\nForecast saved to outputs/results/forecast_2024_2030.csv")

# ---------------------------------------------------------
# 7. Generate Forecast Visualization
# ---------------------------------------------------------
os.makedirs("outputs/figures", exist_ok=True)

fig, ax = plt.subplots(figsize=(14, 6))

# Historical yearly mean SPI
hist_years = yearly['year'].values
hist_spi = yearly['SPI_mean'].values

ax.plot(hist_years, hist_spi, color='blue', marker='o', markersize=6, 
        linewidth=2.5, label='Historical Mean SPI (1981-2023)')

# Connect the historical line to the forecast line (bridge the gap)
bridge_years = [hist_years[-1]] + forecast_years
bridge_spi = [hist_spi[-1]] + ensemble_forecast

# Forecast — bold, solid red line with large markers (same weight as historical)
ax.plot(bridge_years, bridge_spi, color='red', marker='s', markersize=10,
        linewidth=3, linestyle='-', label='Ensemble Forecast (2024-2030)',
        markeredgecolor='darkred', markeredgewidth=1.5, zorder=5)

# Shade forecast region
ax.axvspan(2023.5, 2030.5, alpha=0.10, color='red', label='Forecast Region')

# Drought threshold lines
ax.axhline(0, color='black', linewidth=0.8)
ax.axhline(-1.0, color='orange', linewidth=1.2, linestyle='--', alpha=0.8)
ax.axhline(-1.5, color='red', linewidth=1.2, linestyle='--', alpha=0.8)
ax.text(1982, -1.07, 'Moderate Drought Threshold (-1.0)', fontsize=9, color='orange', fontweight='bold')
ax.text(1982, -1.57, 'Severe Drought Threshold (-1.5)', fontsize=9, color='red', fontweight='bold')

# Annotate forecast values — alternate above/below to avoid clutter
offsets = [(0, 22), (0, -28), (0, 22), (0, -28), (0, 22), (0, -28), (0, 22)]
for i, yr in enumerate(forecast_years):
    ax.annotate(f"{ensemble_forecast[i]:+.2f}",
                (yr, ensemble_forecast[i]),
                textcoords="offset points", xytext=offsets[i],
                ha='center', fontsize=9, fontweight='bold', color='darkred',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', edgecolor='red', alpha=0.9),
                arrowprops=dict(arrowstyle='->', color='red', lw=0.8))

ax.set_xlabel('Year', fontsize=13, fontweight='bold')
ax.set_ylabel('Mean SPI Value (JJAS)', fontsize=13, fontweight='bold')
ax.set_title('Eastern Uttar Pradesh — Drought Forecast to 2030\n(Ensemble of RF, GB, MLP, SVR trained on 1981-2023)', fontsize=14, fontweight='bold')
ax.set_xticks(list(range(1981, 2031, 2)))
ax.tick_params(axis='x', rotation=45, labelsize=10)
ax.tick_params(axis='y', labelsize=10)
ax.grid(linestyle=':', alpha=0.5)
ax.legend(fontsize=11, loc='upper left')
plt.tight_layout()
plt.savefig("outputs/figures/Fig12_Forecast_2030.png", dpi=300)
plt.close()
print("Generated Fig 12: Forecast to 2030")

# ---------------------------------------------------------
# 8. Feature Importance in PCA Space
#    RF operates on 2 PCA components; show both component
#    importances AND the PCA loadings of original features.
# ---------------------------------------------------------

# --- 8a. PCA Component Importances (RF sees 2 dims) ---
pca_importances = rf.feature_importances_          # shape: (2,)
pca_labels = [f"PC{i+1}\n({pca.explained_variance_ratio_[i]*100:.1f}% var)"
              for i in range(len(pca_importances))]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: PCA component importances
axes[0].bar(pca_labels, pca_importances, color=['#2196F3', '#FF9800'],
            edgecolor='black', width=0.5)
axes[0].set_ylabel('Feature Importance (Gini)')
axes[0].set_title('Random Forest Importance\nin PCA Space')
axes[0].set_ylim(0, 1)
for i, v in enumerate(pca_importances):
    axes[0].text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')
axes[0].grid(axis='y', linestyle=':', alpha=0.5)

# Right: PCA Loadings heatmap — contribution of original lag features to each PC
import matplotlib.colors as mcolors
components = pca.components_          # shape: (2, n_features)
# Use only a readable subset: one lag of each base variable
base_feature_cols = [c for c in feature_cols]   # 8 or 9 base variables
loading_labels = [f"{col}\n(lag1)" for col in base_feature_cols]
# Take the loadings for lag-1 features (indices 0..len(feature_cols)-1)
n_base = len(base_feature_cols)
loading_data = components[:, :n_base]           # shape: (2, n_base)

im = axes[1].imshow(loading_data, cmap='RdBu_r', aspect='auto',
                    vmin=-1, vmax=1)
axes[1].set_xticks(range(n_base))
axes[1].set_xticklabels(loading_labels, rotation=45, ha='right', fontsize=8)
axes[1].set_yticks(range(2))
axes[1].set_yticklabels([f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                          f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"])
axes[1].set_title("PCA Loadings\n(lag-1 features → principal components)")
plt.colorbar(im, ax=axes[1], label='Loading weight')

# Annotate each cell
for r in range(2):
    for c in range(n_base):
        axes[1].text(c, r, f"{loading_data[r, c]:.2f}",
                     ha='center', va='center', fontsize=7,
                     color='white' if abs(loading_data[r, c]) > 0.5 else 'black')

plt.suptitle('Feature Importance & PCA Loadings for SPI Forecasting',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/figures/Fig13_Forecast_Feature_Importance.png", dpi=300)
plt.close()
print("Generated Fig 13: Forecast Feature Importance (PCA components + loadings)")

print("\n" + "=" * 60)
print("  Forecast pipeline complete!")
print("=" * 60)
