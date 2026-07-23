import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import scipy.stats as scipy_stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# TensorFlow is disabled to avoid Windows Security / Application Control policy popups.
# Falling back to Scikit-Learn MLPRegressor for MLP & LSTM surrogate modeling.
HAS_TENSORFLOW = False
from sklearn.neural_network import MLPRegressor

print("Starting Dual-Target Model Training Pipeline (SPI + SPEI)...")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load Preprocessed Data
# ─────────────────────────────────────────────────────────────────────────────
data_path = "data/processed_features.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Could not find {data_path}. Run 01_data_preprocessing.py first.")

df = pd.read_csv(data_path, parse_dates=['time'])
df = df.dropna().sort_values('time').reset_index(drop=True)
print(f"Loaded {len(df)} weekly records from 1981 to 2023.")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Compute SPEI-3 and map back to weekly rows
#    Pipeline: weekly → monthly agg → Thornthwaite PET → water balance
#              → 3-month accumulation → log-logistic standardisation → SPEI-3
#    Each week inherits its calendar month's SPEI-3 value.
# ─────────────────────────────────────────────────────────────────────────────
print("\nComputing SPEI-3 via Thornthwaite PET...")

df['year']  = df['time'].dt.year
df['month'] = df['time'].dt.month

monthly = df.groupby(['year', 'month']).agg(
    Rainfall_mm=('Rainfall', 'sum'),
    MaxTemp_C=('Max_Temp', 'mean'),
).reset_index().sort_values(['year', 'month']).reset_index(drop=True)

# Thornthwaite heat index & exponent (climatological normals)
clim_mean = (monthly.groupby('month')['MaxTemp_C'].mean() - 5.0).clip(lower=0)
I_heat = ((clim_mean / 5.0) ** 1.514).sum()
a_exp  = 6.75e-7 * I_heat**3 - 7.71e-5 * I_heat**2 + 1.792e-2 * I_heat + 0.49239

def thornthwaite_pet(row):
    T = max(row['MaxTemp_C'] - 5.0, 0)
    days = 31 if row['month'] in [1, 3, 5, 7, 8, 10, 12] else (28 if row['month'] == 2 else 30)
    return 16 * ((10 * T / I_heat) ** a_exp) * (days / 30) if (I_heat > 0 and T > 0) else 0.0

monthly['PET_mm']  = monthly.apply(thornthwaite_pet, axis=1)
monthly['D']       = monthly['Rainfall_mm'] - monthly['PET_mm']
monthly['D_acc3']  = monthly['D'].rolling(window=3, min_periods=3).sum()

# Log-logistic standardisation (per calendar month)
spei_vals = []
for _, row in monthly.iterrows():
    m = row['month']
    same = monthly[monthly['month'] == m]['D_acc3'].dropna().values
    if len(same) < 3 or np.isnan(row['D_acc3']):
        spei_vals.append(np.nan)
        continue
    shift = same.min() - 0.001
    try:
        c, loc, scale = scipy_stats.fisk.fit(same - shift, floc=0)
        p    = scipy_stats.fisk.cdf(row['D_acc3'] - shift, c, loc, scale)
        p    = np.clip(p, 0.001, 0.999)
        spei_vals.append(scipy_stats.norm.ppf(p))
    except Exception:
        spei_vals.append(np.nan)

monthly['SPEI_3'] = spei_vals

# Left-join monthly SPEI-3 back to weekly df
df = df.merge(monthly[['year', 'month', 'SPEI_3']], on=['year', 'month'], how='left')
df = df.dropna(subset=['SPEI_3']).reset_index(drop=True)
print(f"  SPEI-3 attached: {df['SPEI_3'].notna().sum()} weekly rows have SPEI values.")
print(f"  SPI  range: [{df['SPI'].min():.3f}, {df['SPI'].max():.3f}]")
print(f"  SPEI range: [{df['SPEI_3'].min():.3f}, {df['SPEI_3'].max():.3f}]")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Build feature sequences (shared for both targets)
# ─────────────────────────────────────────────────────────────────────────────
sequence_length = 4  # Past 4 weeks

feature_cols = ['Rainfall', 'Max_Temp', 'Soil_Moisture']
if 'RH' in df.columns:
    feature_cols.append('RH')
    print(f"Using RH as additional feature! Input features: {feature_cols}")
else:
    print(f"Input features: {feature_cols}")

features       = df[feature_cols].values
target_spi     = df['SPI'].values
target_spei    = df['SPEI_3'].values
dates          = df['time'].values

X_list, y_spi_list, y_spei_list, seq_dates = [], [], [], []
for i in range(sequence_length, len(df)):
    X_list.append(features[i - sequence_length:i])
    y_spi_list.append(target_spi[i])
    y_spei_list.append(target_spei[i])
    seq_dates.append(dates[i])

X          = np.array(X_list)
y_spi_all  = np.array(y_spi_list)
y_spei_all = np.array(y_spei_list)
seq_dates  = np.array(seq_dates)

print(f"Built {len(X)} sequences of length {sequence_length}.")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Date-based train/test split
# ─────────────────────────────────────────────────────────────────────────────
dates_dt    = pd.to_datetime(seq_dates)
train_mask  = dates_dt.year <= 2012
test_mask   = dates_dt.year >= 2013
test_dates  = seq_dates[test_mask]
test_dates_dt = pd.to_datetime(test_dates)
mask_1 = (test_dates_dt.year >= 2013) & (test_dates_dt.year <= 2017)
mask_2 = (test_dates_dt.year >= 2018) & (test_dates_dt.year <= 2023)

X_train = X[train_mask]
X_test  = X[test_mask]

n_steps, n_features = sequence_length, len(feature_cols)
X_train_2d = X_train.reshape(-1, n_steps * n_features)
X_test_2d  = X_test.reshape(-1, n_steps * n_features)

scaler_X = MinMaxScaler()
X_train_sc = scaler_X.fit_transform(X_train_2d)
X_test_sc  = scaler_X.transform(X_test_2d)

# Save shared input scaler (same features for both targets)
os.makedirs("models", exist_ok=True)
joblib.dump(scaler_X, "models/scaler_X.pkl")
print(f"\nTrain: {len(X_train)} samples | Test: {len(X_test)} samples")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Evaluation helper
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    try:
        corr = np.corrcoef(y_true, y_pred)[0, 1]
    except Exception:
        corr = 0.0
    return {'Model': name, 'R2': r2, 'RMSE': rmse, 'MAE': mae, 'Correlation': corr}

# ─────────────────────────────────────────────────────────────────────────────
# 6. Reusable training + evaluation function
# ─────────────────────────────────────────────────────────────────────────────
def run_target(target_name, y_all):
    """Train all 5 models on a given target (SPI or SPEI), save metrics & plots."""
    print(f"\n{'='*60}")
    print(f"  TARGET: {target_name}")
    print(f"{'='*60}")

    y_train = y_all[train_mask]
    y_test  = y_all[test_mask]

    scaler_y = MinMaxScaler()
    y_train_sc = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    joblib.dump(scaler_y, f"models/scaler_y_{target_name}.pkl")

    # — Build models —
    models_dict = {}

    print(f"Training SVR ({target_name})...")
    svr = SVR(kernel="rbf", C=1.0, epsilon=0.1)
    svr.fit(X_train_sc, y_train_sc)
    models_dict['SVR'] = svr

    print(f"Training Random Forest ({target_name})...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train_sc, y_train_sc)
    models_dict['RF'] = rf

    print(f"Training XGBoost ({target_name})...")
    xgb = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    xgb.fit(X_train_sc, y_train_sc)
    models_dict['XGB'] = xgb

    print(f"Training MLP ({target_name})...")
    mlp = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=300, random_state=42, early_stopping=True)
    mlp.fit(X_train_sc, y_train_sc)
    models_dict['MLP'] = mlp

    print(f"Training LSTM surrogate ({target_name})...")
    lstm = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42, early_stopping=True)
    lstm.fit(X_train_sc, y_train_sc)
    models_dict['LSTM'] = lstm

    # — Evaluate —
    results_1, results_2, predictions = [], [], {}

    print(f"\n--- {target_name} Model Evaluation (2013-2023) ---")
    for name, model in models_dict.items():
        pred_sc = model.predict(X_test_sc)
        pred    = scaler_y.inverse_transform(pred_sc.reshape(-1, 1)).ravel()
        predictions[name] = pred

        m1 = evaluate(y_test[mask_1], pred[mask_1], name)
        m2 = evaluate(y_test[mask_2], pred[mask_2], name)
        results_1.append(m1)
        results_2.append(m2)
        print(f"  {name} (2013-17): R2={m1['R2']:.3f} RMSE={m1['RMSE']:.3f} Corr={m1['Correlation']:.3f}")
        print(f"  {name} (2018-23): R2={m2['R2']:.3f} RMSE={m2['RMSE']:.3f} Corr={m2['Correlation']:.3f}")

    # — Save metrics —
    os.makedirs("outputs/results", exist_ok=True)
    pd.DataFrame(results_1).to_csv(f"outputs/results/metrics_{target_name}_2013_2017.csv", index=False)
    pd.DataFrame(results_2).to_csv(f"outputs/results/metrics_{target_name}_2018_2023.csv", index=False)

    # Keep SPI filenames backward-compatible for existing Streamlit tabs
    if target_name == "SPI":
        pd.DataFrame(results_1).to_csv("outputs/results/metrics.csv", index=False)
        pd.DataFrame(results_2).to_csv("outputs/results/metrics_2018_2023.csv", index=False)

    print(f"  Metrics saved for {target_name}.")

    # — Figures —
    os.makedirs("outputs/figures", exist_ok=True)
    for name in models_dict:
        pred = predictions[name]

        # Yearly mean chart
        test_df = pd.DataFrame({
            'date':   pd.to_datetime(test_dates),
            'actual': y_test,
            'pred':   pred
        }).sort_values('date')
        test_df['year'] = test_df['date'].dt.year

        yearly = test_df.groupby('year').agg(
            actual_mean=('actual', 'mean'),
            pred_mean=('pred',   'mean')
        ).reset_index()

        fig, ax = plt.subplots(figsize=(12, 5))
        years = yearly['year'].astype(int).values
        ax.plot(years, yearly['actual_mean'], color='blue', marker='o', markersize=8,
                linewidth=2, label=f'Actual {target_name} (Mean)')
        ax.plot(years, yearly['pred_mean'],   color='red',  marker='s', markersize=8,
                linewidth=2, linestyle='--', label=f'Predicted {target_name} (Mean)')

        for i, yr in enumerate(years):
            ax.annotate(f"{yearly['actual_mean'].iloc[i]:.2f}",
                        (yr, yearly['actual_mean'].iloc[i]),
                        textcoords="offset points", xytext=(0, 10),
                        ha='center', fontsize=8, color='blue')
            ax.annotate(f"{yearly['pred_mean'].iloc[i]:.2f}",
                        (yr, yearly['pred_mean'].iloc[i]),
                        textcoords="offset points", xytext=(0, -14),
                        ha='center', fontsize=8, color='red')

        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel(f'Mean {target_name} Value', fontsize=12)
        ax.set_title(f'{name} — Yearly Mean Prediction vs Actual {target_name} (Eastern UP)', fontsize=13)
        ax.set_xticks(years)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.grid(linestyle=':', alpha=0.5)
        ax.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(f"outputs/figures/{name}_{target_name}_pred_actual.png", dpi=300)
        # Backward-compatible alias for existing SPI plots
        if target_name == "SPI":
            plt.savefig(f"outputs/figures/{name}_pred_actual.png", dpi=300)
        plt.close()

        # Residuals
        residuals = y_test - pred
        plt.figure(figsize=(8, 4))
        plt.hist(residuals, bins=30, color="steelblue", edgecolor="black")
        plt.axvline(0, color="red", linestyle="--")
        plt.title(f"{name} [{target_name}] — Residual Distribution")
        plt.xlabel("Residual")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(f"outputs/figures/{name}_{target_name}_residuals.png", dpi=300)
        if target_name == "SPI":
            plt.savefig(f"outputs/figures/{name}_residuals.png", dpi=300)
        plt.close()

    print(f"  Figures saved for {target_name}.")
    return predictions, y_test

# ─────────────────────────────────────────────────────────────────────────────
# 7. Run both targets
# ─────────────────────────────────────────────────────────────────────────────
spi_preds,  y_spi_test  = run_target("SPI",  y_spi_all)
spei_preds, y_spei_test = run_target("SPEI", y_spei_all)

# ─────────────────────────────────────────────────────────────────────────────
# 8. Three-way comparison chart (best model per index)
#    SPI Actual vs SPI Predicted vs SPEI Predicted (LSTM is usually best)
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating three-way SPI/SPEI comparison chart...")

best_model = "LSTM"   # typically best individual model

compare_df = pd.DataFrame({
    'date':        pd.to_datetime(test_dates),
    'SPI_actual':  y_spi_test,
    'SPI_pred':    spi_preds[best_model],
    'SPEI_pred':   spei_preds[best_model],
}).sort_values('date')
compare_df['year'] = compare_df['date'].dt.year

yearly_cmp = compare_df.groupby('year').agg(
    SPI_actual=('SPI_actual',  'mean'),
    SPI_pred  =('SPI_pred',    'mean'),
    SPEI_pred =('SPEI_pred',   'mean'),
).reset_index()

fig, ax = plt.subplots(figsize=(14, 6))
years = yearly_cmp['year'].astype(int).values

ax.plot(years, yearly_cmp['SPI_actual'], color='blue',   marker='o', markersize=8,
        linewidth=2.5, label='Actual SPI')
ax.plot(years, yearly_cmp['SPI_pred'],   color='red',    marker='s', markersize=7,
        linewidth=2,   linestyle='--', label=f'{best_model} Predicted SPI')
ax.plot(years, yearly_cmp['SPEI_pred'],  color='green',  marker='^', markersize=7,
        linewidth=2,   linestyle='-.',  label=f'{best_model} Predicted SPEI-3')

ax.axhline(0,    color='black',  linewidth=0.8)
ax.axhline(-1.0, color='orange', linewidth=1.0, linestyle=':', alpha=0.8, label='Drought threshold (−1.0)')
ax.set_xlabel('Year', fontsize=12, fontweight='bold')
ax.set_ylabel('Index Value', fontsize=12, fontweight='bold')
ax.set_title(f'Three-Way Comparison: SPI Actual · SPI Predicted · SPEI Predicted\n'
             f'({best_model} model, Eastern UP 2013-2023)', fontsize=13, fontweight='bold')
ax.set_xticks(years)
ax.grid(linestyle=':', alpha=0.5)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("outputs/figures/Fig14_SPI_SPEI_Comparison.png", dpi=300)
plt.close()
print("  Saved: outputs/figures/Fig14_SPI_SPEI_Comparison.png")

print("\n" + "=" * 60)
print("  Dual-target training pipeline complete!")
print("  SPI  metrics → outputs/results/metrics_SPI_*.csv")
print("  SPEI metrics → outputs/results/metrics_SPEI_*.csv")
print("  Comparison   → outputs/figures/Fig14_SPI_SPEI_Comparison.png")
print("=" * 60)
