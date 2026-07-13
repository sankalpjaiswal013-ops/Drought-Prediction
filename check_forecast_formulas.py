import pandas as pd
import numpy as np

# Load processed weekly data
try:
    df = pd.read_csv("data/processed_features.csv", parse_dates=['time'])
    df['year'] = df['time'].dt.year
    
    # Calculate yearly stats
    yearly = df.groupby('year').agg(
        SPI_mean=('SPI', 'mean'),
        Rainfall_total=('Rainfall', 'sum')
    ).reset_index()
    
    # Calculate historical average yearly total rainfall
    mean_yearly_rf = yearly['Rainfall_total'].mean()
    
    # Calculate actual percentage deviation of yearly rainfall
    yearly['RF_pct_dev'] = ((yearly['Rainfall_total'] - mean_yearly_rf) / mean_yearly_rf) * 100
    
    # Find linear relationship: RF_pct_dev = alpha * SPI_mean
    # We do a simple least-squares fit (no intercept, because when SPI_mean = 0, RF_pct_dev should be 0)
    alpha = np.sum(yearly['SPI_mean'] * yearly['RF_pct_dev']) / np.sum(yearly['SPI_mean'] ** 2)
    
    # Fit with intercept for comparison
    slope, intercept = np.polyfit(yearly['SPI_mean'], yearly['RF_pct_dev'], 1)
    
    print("--- Historical Rainfall & SPI Calibration (1981-2023) ---")
    print(f"Mean Historical Yearly Total Rainfall: {mean_yearly_rf:.2f} mm")
    print(f"Calibrated Multiplier (No Intercept): {alpha:.4f}")
    print(f"Calibrated Multiplier (With Intercept): Slope = {slope:.4f}, Intercept = {intercept:.4f}")
    
    # Test on some historical years
    print("\nComparison for selected historical years:")
    for yr in [1987, 2002, 2015, 2018, 2023]:
        row = yearly[yearly['year'] == yr].iloc[0]
        cal_dev = row['SPI_mean'] * alpha
        print(f"  Year {yr}: SPI_mean = {row['SPI_mean']:+.3f} | Actual RF Dev = {row['RF_pct_dev']:+.2f}% | Calibrated Est = {cal_dev:+.2f}%")
        
    # Load future forecast
    if pd.io.common.file_exists("outputs/results/forecast_2024_2030.csv"):
        fc = pd.read_csv("outputs/results/forecast_2024_2030.csv")
        print("\n--- Future Forecast Predictions (2024-2030) ---")
        for idx, row in fc.iterrows():
            year = int(row['Year'])
            spi = row['Ensemble_SPI']
            est_rf_dev = spi * alpha
            print(f"  Year {year}: Ensemble SPI = {spi:+.3f} -> Predicted Rainfall Deviation = {est_rf_dev:+.2f}%")
except Exception as e:
    print(f"Error: {e}")
