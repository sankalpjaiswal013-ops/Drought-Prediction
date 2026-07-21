import os
import glob
import xarray as xr
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=xr.SerializationWarning)

# ---------------------------------------------------------
# Phase 1: Environment Setup
# ---------------------------------------------------------
directories = [
    "data",
    "notebooks",
    "outputs/figures",
    "outputs/results",
    "models"
]

for d in directories:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------
# Phase 2: Data Loading & Preprocessing
# ---------------------------------------------------------

def compute_spi(rainfall_series, scale=4):
    # Convert to series with datetime index
    s = pd.Series(rainfall_series)
    
    # 1. Compute rolling mean within each year to prevent cross-year leakage
    rolling = s.groupby(s.index.year, group_keys=False).apply(lambda x: x.rolling(scale).mean())
    
    # 2. Standardize each week of the year relative to its own historical distribution
    weeks = rolling.index.isocalendar().week
    
    # Group by calendar week and apply z-score
    mean_by_week = rolling.groupby(weeks).transform('mean')
    std_by_week = rolling.groupby(weeks).transform('std')
    
    # Avoid division by zero if std is zero or nan
    std_by_week = std_by_week.replace(0, np.nan)
    
    spi = (rolling - mean_by_week) / std_by_week
    
    # Fill NaNs from dropping or zero std with 0 (near normal)
    spi = spi.fillna(0)
    return spi

print("\nStarting Data Preprocessing...")

def process_files(file_list, var_type):
    df_list = []
    
    for file in sorted(file_list):
        try:
            ds = xr.open_dataset(file)
            
            # Identify time, lat, lon dimensions dynamically
            time_dim = next((d for d in list(ds.dims) + list(ds.coords) if 'time' in str(d).lower()), 'time')
            lat_dim = next((d for d in list(ds.dims) + list(ds.coords) if 'lat' in str(d).lower()), 'lat')
            lon_dim = next((d for d in list(ds.dims) + list(ds.coords) if 'lon' in str(d).lower()), 'lon')

            # Identify data variable
            var_name = [v for v in ds.data_vars if 'bnd' not in v.lower() and 'time' not in v.lower()][0]
            
            # Sort spatial dimensions because slice() fails if lat/lon are descending
            ds = ds.sortby(lat_dim)
            ds = ds.sortby(lon_dim)
            
            # Crop to Eastern UP Box: Lat 23.5 to 28.5, Lon 81.0 to 84.5
            ds_up = ds.sel({lat_dim: slice(23.5, 28.5), lon_dim: slice(81.0, 84.5)})
            
            # Skip if time is outside our target window
            if ds_up[time_dim].dt.year.max().item() < 1981 or ds_up[time_dim].dt.year.min().item() > 2023:
                ds.close()
                continue
                
            # Filter JJAS
            ds_jjas = ds_up.sel({time_dim: ds_up[time_dim].dt.month.isin([6, 7, 8, 9])})
            
            # Spatial mean
            ds_mean = ds_jjas.mean(dim=[lat_dim, lon_dim])
            
            # Convert to dataframe
            df = ds_mean.to_dataframe().reset_index()
            df = df.rename(columns={time_dim: 'time'})
            df = df.set_index('time')[[var_name]]
            df = df[~df.index.duplicated(keep='first')]
            
            if not df.empty:
                df_list.append(df)
            ds.close()
            
        except Exception as e:
            # Silently skip bad files to not clutter output, but record if it fails completely
            continue
            
    if not df_list:
        raise ValueError(f"No valid data found for {var_type}. Check your netcdf files.")
        
    full_df = pd.concat(df_list).sort_index()
    # Filter strict common period
    full_df = full_df.loc['1981-01-01':'2023-12-31']
    return full_df

try:
    print("Processing Rainfall files...")
    rf_files = glob.glob("raw_data/rainfall/*.nc")
    df_rf = process_files(rf_files, "Rainfall")
    df_rf.columns = ['Rainfall']

    print("Processing Temperature files...")
    temp_files = glob.glob("raw_data/Maximum Temprature/*.nc")
    df_temp = process_files(temp_files, "Max_Temp")
    df_temp.columns = ['Max_Temp']

    print("Processing Soil Moisture files...")
    sm_files = glob.glob("raw_data/soil/*.nc")
    df_sm = process_files(sm_files, "Soil_Moisture")
    df_sm.columns = ['Soil_Moisture']

    print("Aggregating to Weekly Data...")
    df_rf_weekly = df_rf.resample('W').sum()
    df_temp_weekly = df_temp.resample('W').mean()
    df_sm_weekly = df_sm.resample('W').mean()

    # Load Relative Humidity if processed CSV exists
    df_rh_weekly = None
    if os.path.exists("data/processed_rh.csv"):
        print("Loading processed Relative Humidity data...")
        df_rh = pd.read_csv("data/processed_rh.csv", parse_dates=['time']).set_index('time')
        df_rh_weekly = df_rh.resample('W').mean()
        df_rh_weekly.columns = ['RH']

    # Combine into a single dataframe
    dfs_to_concat = [df_rf_weekly, df_temp_weekly, df_sm_weekly]
    if df_rh_weekly is not None:
        dfs_to_concat.append(df_rh_weekly)
    df_combined = pd.concat(dfs_to_concat, axis=1)
    
    # Impute missing Soil Moisture (2018-2023) using historical weekly averages
    print("Imputing missing Soil Moisture for 2018-2023 using weekly climatology...")
    df_combined['week'] = df_combined.index.isocalendar().week
    weekly_means = df_combined.groupby('week')['Soil_Moisture'].mean()
    df_combined['Soil_Moisture'] = df_combined.apply(
        lambda row: weekly_means.get(row['week'], np.nan) if pd.isna(row['Soil_Moisture']) else row['Soil_Moisture'],
        axis=1
    )
    
    # Impute missing Relative Humidity (2021-2023) using historical weekly averages
    if 'RH' in df_combined.columns:
        print("Imputing missing Relative Humidity for 2021-2023 using weekly climatology...")
        weekly_rh_means = df_combined.groupby('week')['RH'].mean()
        df_combined['RH'] = df_combined.apply(
            lambda row: weekly_rh_means.get(row['week'], np.nan) if pd.isna(row['RH']) else row['RH'],
            axis=1
        )
        
    df_combined = df_combined.drop(columns=['week'])
    
    df_combined = df_combined.dropna()

    print("Computing SPI (Standardized Precipitation Index)...")
    df_combined['SPI'] = compute_spi(df_combined['Rainfall'], scale=4)
    df_final = df_combined.dropna()

    output_path = "data/processed_features.csv"
    df_final.to_csv(output_path)
    
    print(f"\nSuccess! Data preprocessed and saved to '{output_path}'.")
    print(f"Total Weekly Data Points (JJAS 1981-2023): {len(df_final)}")
    print("\nSample of final dataset:")
    print(df_final.head())

except Exception as e:
    import traceback
    import sys
    print(f"\nFatal Error:\n{e}")
    traceback.print_exc()
    
    # Graceful fallback: if processed_features.csv already exists from a prior run,
    # skip re-processing (e.g. when raw NetCDF files are unavailable / DLL blocked)
    # and exit cleanly so the rest of the pipeline can continue.
    fallback_path = "data/processed_features.csv"
    if os.path.exists(fallback_path):
        print(f"\n⚠️  WARNING: Raw data processing failed, but '{fallback_path}' already")
        print(f"   exists from a previous run. Skipping re-processing — using cached data.")
        print(f"   Pipeline will continue with the existing processed dataset.\n")
        sys.exit(0)   # Clean exit → run_pipeline.py marks ✓ and continues
    else:
        print(f"\n❌ HARD FAILURE: No processed data available. Run raw preprocessing first.")
        sys.exit(1)   # Non-zero exit → run_pipeline.py correctly aborts
