import pandas as pd

# Load historical data
df_hist = pd.read_csv("data/processed_features.csv", parse_dates=['time'])
x_bar = df_hist['SPI'].mean()
print(f"X_bar = {x_bar}")

# Load forecast
forecast_df = pd.read_csv("outputs/results/forecast_2024_2030.csv")
x_2026 = forecast_df.loc[forecast_df['Year'] == 2026, 'Ensemble_SPI'].values[0]

pct = ((x_2026 - x_bar) / x_bar) * 100
print(f"2026 SPI = {x_2026}")
print(f"Percentage using exact formula = {pct}%")

# Maybe they mean Rainfall percentage deviation instead of SPI? 
# Because SPI is already standard deviation from the mean, it doesn't make physical sense to take the % deviation of a z-score relative to its mean (which is zero).
