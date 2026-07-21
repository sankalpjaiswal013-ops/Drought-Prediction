"""
recompute_spi.py
----------------
Recomputes the SPI column in data/processed_features.csv using the
scientifically corrected formula:
  1. Rolling mean computed WITHIN each year (prevents cross-year leakage)
  2. Z-score computed PER calendar week (removes seasonal cycle bias)

Run this whenever the raw NetCDF reprocessing is unavailable (e.g. DLL
blocked) but you want to update the SPI calculation from cached data.
"""

import pandas as pd
import numpy as np

print("=" * 60)
print("  Recomputing SPI from cached processed_features.csv")
print("=" * 60)

# ── Load cached data ───────────────────────────────────────────
path = "data/processed_features.csv"
df = pd.read_csv(path, parse_dates=['time'])
df = df.set_index('time').sort_index()

print(f"\nLoaded {len(df)} weekly records ({df.index.year.min()}–{df.index.year.max()})")
print(f"Columns: {list(df.columns)}")

# ── Corrected compute_spi ──────────────────────────────────────
def compute_spi_corrected(rainfall_series, scale=4):
    """
    Scientifically correct SPI calculation:
    - Rolling mean is computed PER YEAR (no cross-year leakage)
    - Z-score is computed PER CALENDAR WEEK (removes seasonal bias)
    """
    s = pd.Series(rainfall_series)

    # Step 1: Within-year rolling mean
    rolling = s.groupby(s.index.year, group_keys=False).apply(
        lambda x: x.rolling(scale, min_periods=1).mean()
    )

    # Step 2: Standardize each calendar week against its own climatology
    weeks = rolling.index.isocalendar().week
    mean_by_week = rolling.groupby(weeks).transform('mean')
    std_by_week  = rolling.groupby(weeks).transform('std')
    std_by_week  = std_by_week.replace(0, np.nan)

    spi = (rolling - mean_by_week) / std_by_week

    # NaN → 0 (near normal) for weeks with insufficient history
    spi = spi.fillna(0)
    return spi

# ── Apply new formula ──────────────────────────────────────────
print("\nApplying corrected SPI formula...")
old_spi = df['SPI'].copy()
df['SPI'] = compute_spi_corrected(df['Rainfall'], scale=4)

# ── Quick validation ───────────────────────────────────────────
yearly = df.groupby(df.index.year).agg(
    SPI_old  = pd.NamedAgg(column='SPI',      aggfunc='mean'),  # will be new
    Rain_tot = pd.NamedAgg(column='Rainfall', aggfunc='sum'),
)
yearly['SPI_old_val'] = old_spi.groupby(old_spi.index.year).mean()
yearly['SPI_new_val'] = df['SPI'].groupby(df.index.year).mean()

print(f"\n{'Year':>6} {'Old SPI Mean':>13} {'New SPI Mean':>13} {'Δ':>8}")
print("-" * 45)

known_years = {1983, 1987, 1994, 2002, 2004, 2009, 2014, 2015, 2019, 2020}
for yr, row in yearly.iterrows():
    marker = " ◄" if yr in known_years else ""
    print(f"{int(yr):>6} {row['SPI_old_val']:>+13.4f} {row['SPI_new_val']:>+13.4f} "
          f"{row['SPI_new_val']-row['SPI_old_val']:>+8.4f}{marker}")

# ── Save back ──────────────────────────────────────────────────
df = df.reset_index()
df.to_csv(path, index=False)
print(f"\n✅ Saved updated SPI to '{path}'")
print("   Now run: python sanity_check_droughts.py")
print("=" * 60)
