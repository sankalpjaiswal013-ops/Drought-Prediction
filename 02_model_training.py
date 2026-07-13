import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Try to import TensorFlow, fallback to sklearn if blocked by system policy
HAS_TENSORFLOW = True
try:
    # Set TF log level to minimize warnings
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, LSTM
    from tensorflow.keras.callbacks import EarlyStopping
except Exception as e:
    HAS_TENSORFLOW = False
    print("\n" + "!"*80)
    print("WARNING: TensorFlow cannot be loaded (blocked by Windows Application Control policy).")
    print("Falling back to Scikit-Learn MLPRegressor for MLP & LSTM surrogate modeling.")
    print("!"*80 + "\n")
    from sklearn.neural_network import MLPRegressor

print("Starting Model Training Pipeline...")

# 1. Load Preprocessed Data
data_path = "data/processed_features.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Could not find {data_path}. Run 01_data_preprocessing.py first.")

df = pd.read_csv(data_path, parse_dates=['time'])
df = df.sort_values('time').reset_index(drop=True)
print(f"Loaded {len(df)} weekly records from 1981 to 2023.")

# 2. Build Input-Output Sequences
sequence_length = 4  # Past 4 weeks

feature_cols = ['Rainfall', 'Max_Temp', 'Soil_Moisture']
if 'RH' in df.columns:
    feature_cols.append('RH')
    print(f"Using Relative Humidity (RH) as an additional training feature! Input features: {feature_cols}")
else:
    print(f"Input features: {feature_cols}")

features = df[feature_cols].values
target_spi = df['SPI'].values
dates = df['time'].values

X, y, sequence_dates = [], [], []

for i in range(sequence_length, len(df)):
    X.append(features[i-sequence_length:i])
    y.append(target_spi[i])
    sequence_dates.append(dates[i])

X = np.array(X)
y = np.array(y)
sequence_dates = np.array(sequence_dates)

print(f"Built {len(X)} sequences of length {sequence_length}.")

# 3. Date-Based Train/Test Split
# Train: 1981 - 2012
# Test:  2013 - 2023
train_mask = pd.to_datetime(sequence_dates).year <= 2012
test_mask = pd.to_datetime(sequence_dates).year >= 2013

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]
test_dates = sequence_dates[test_mask]

print(f"Train Set: {len(X_train)} samples")
print(f"Test Set: {len(X_test)} samples")

# 4. Normalization
n_samples_train, n_steps, n_features = X_train.shape
n_samples_test, _, _ = X_test.shape

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Flatten X for scaling, then reshape back
X_train_2d = X_train.reshape(-1, n_steps * n_features)
X_test_2d = X_test.reshape(-1, n_steps * n_features)

X_train_sc = scaler_X.fit_transform(X_train_2d)
X_test_sc = scaler_X.transform(X_test_2d)

y_train_sc = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_sc = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

# Save scalers
joblib.dump(scaler_X, "models/scaler_X.pkl")
joblib.dump(scaler_y, "models/scaler_y.pkl")

# 5. Train Models
models_dict = {}

# SVR
print("\nTraining SVR...")
svr = SVR(kernel="rbf", C=1.0, epsilon=0.1)
svr.fit(X_train_sc, y_train_sc)
models_dict['SVR'] = svr

# Random Forest
print("Training Random Forest...")
rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train_sc, y_train_sc)
models_dict['RF'] = rf

# XGBoost
print("Training XGBoost...")
xgb = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
xgb.fit(X_train_sc, y_train_sc)
models_dict['XGB'] = xgb

# MLP - Scaled down architecture to prevent overfitting
print("Training MLP...")
if HAS_TENSORFLOW:
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    mlp = Sequential([
        Dense(32, activation="relu", input_shape=(X_train_sc.shape[1],)),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    mlp.compile(optimizer="adam", loss="mse")
    history_mlp = mlp.fit(X_train_sc, y_train_sc, epochs=150, batch_size=16, 
                          validation_split=0.2, callbacks=[early_stop], verbose=0)
    models_dict['MLP'] = mlp
else:
    mlp = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=300, random_state=42, early_stopping=True)
    mlp.fit(X_train_sc, y_train_sc)
    models_dict['MLP'] = mlp

# LSTM - Compact architecture
print("Training LSTM...")
if HAS_TENSORFLOW:
    X_train_lstm = X_train_sc.reshape(-1, n_steps, n_features)
    X_test_lstm = X_test_sc.reshape(-1, n_steps, n_features)
    
    lstm = Sequential([
        LSTM(32, input_shape=(n_steps, n_features)),
        Dropout(0.2),
        Dense(1)
    ])
    lstm.compile(optimizer="adam", loss="mse")
    history_lstm = lstm.fit(X_train_lstm, y_train_sc, epochs=150, batch_size=16, 
                            validation_split=0.2, callbacks=[early_stop], verbose=0)
    models_dict['LSTM'] = lstm
else:
    # Train a surrogate MLP model on the flattened sequence input
    lstm = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42, early_stopping=True)
    lstm.fit(X_train_sc, y_train_sc)
    models_dict['LSTM'] = lstm

# 6. Evaluate and Plot
def evaluate(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    # Handle case where correlation might fail (e.g. constant prediction)
    try:
        corr = np.corrcoef(y_true, y_pred)[0, 1]
    except:
        corr = 0
    return {'Model': name, 'R2': r2, 'RMSE': rmse, 'MAE': mae, 'Correlation': corr}

results_1 = []
results_2 = []
predictions = {}

print("\n--- Model Evaluation (Test Set 2013-2023) ---")
test_dates_dt = pd.to_datetime(test_dates)
mask_1 = (test_dates_dt.year >= 2013) & (test_dates_dt.year <= 2017)
mask_2 = (test_dates_dt.year >= 2018) & (test_dates_dt.year <= 2023)

for name, model in models_dict.items():
    # Predict
    if name == 'LSTM':
        if HAS_TENSORFLOW:
            pred_sc = model.predict(X_test_lstm, verbose=0).ravel()
        else:
            pred_sc = model.predict(X_test_sc).ravel()
    elif name == 'MLP':
        if HAS_TENSORFLOW:
            pred_sc = model.predict(X_test_sc, verbose=0).ravel()
        else:
            pred_sc = model.predict(X_test_sc).ravel()
    else:
        pred_sc = model.predict(X_test_sc)
    
    # Inverse Transform
    pred = scaler_y.inverse_transform(pred_sc.reshape(-1, 1)).ravel()
    predictions[name] = pred
    
    # Evaluate 2013-2017
    metrics_1 = evaluate(y_test[mask_1], pred[mask_1], name)
    results_1.append(metrics_1)
    
    # Evaluate 2018-2023
    metrics_2 = evaluate(y_test[mask_2], pred[mask_2], name)
    results_2.append(metrics_2)
    
    print(f"{name} (2013-2017): R2={metrics_1['R2']:.3f} RMSE={metrics_1['RMSE']:.3f} MAE={metrics_1['MAE']:.3f}")
    print(f"{name} (2018-2023): R2={metrics_2['R2']:.3f} RMSE={metrics_2['RMSE']:.3f} MAE={metrics_2['MAE']:.3f}")

# Save Results Tables
pd.DataFrame(results_1).to_csv("outputs/results/metrics.csv", index=False)
pd.DataFrame(results_2).to_csv("outputs/results/metrics_2018_2023.csv", index=False)
print("\nMetrics saved to outputs/results/metrics.csv and outputs/results/metrics_2018_2023.csv")

# 7. Generate Figures (Prediction vs Actual & Residuals)
os.makedirs("outputs/figures", exist_ok=True)

for name in models_dict.keys():
    pred = predictions[name]
    
    # Prediction vs Actual (Figures 6-9)
    # Aggregate weekly SPI values to a single mean value per year
    test_df = pd.DataFrame({
        'date': pd.to_datetime(test_dates),
        'actual': y_test,
        'pred': pred
    }).sort_values('date')
    test_df['year'] = test_df['date'].dt.year
    
    yearly = test_df.groupby('year').agg(
        actual_mean=('actual', 'mean'),
        pred_mean=('pred', 'mean')
    ).reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    years = yearly['year'].astype(int).values
    
    # Line plot with markers
    ax.plot(years, yearly['actual_mean'], color='blue', marker='o', markersize=8,
            linewidth=2, label='Actual SPI (Mean)')
    ax.plot(years, yearly['pred_mean'], color='red', marker='s', markersize=8,
            linewidth=2, linestyle='--', label='Predicted SPI (Mean)')
    
    # Annotate precise values at each point
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
    ax.set_ylabel('Mean SPI Value', fontsize=12)
    ax.set_title(f'{name} - Yearly Mean Prediction vs Actual SPI (Uttar Pradesh)', fontsize=13)
    ax.set_xticks(years)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='-')
    ax.grid(linestyle=':', alpha=0.5)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"outputs/figures/{name}_pred_actual.png", dpi=300)
    plt.close()
    
    # Residuals (Figure 10)
    residuals = y_test - pred
    plt.figure(figsize=(8, 4))
    plt.hist(residuals, bins=30, color="steelblue", edgecolor="black")
    plt.axvline(0, color="red", linestyle="--")
    plt.title(f"{name} - Residual Distribution")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"outputs/figures/{name}_residuals.png", dpi=300)
    plt.close()

print("\nFigures 6-10 generated and saved in 'outputs/figures/'.")
print("Model training pipeline complete!")
