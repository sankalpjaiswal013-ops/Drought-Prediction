import xarray as xr
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def check_time_alignment(datasets: dict) -> bool:
    """
    Checks if all provided datasets have exactly the same time coordinates.
    
    Args:
        datasets: Dictionary mapping dataset names to xarray.Dataset objects.
    """
    logging.info("Running Time-Alignment Check...")
    
    if not datasets:
        logging.error("No datasets provided.")
        return False

    names = list(datasets.keys())
    reference_name = names[0]
    reference_time = datasets[reference_name]['time'].values

    all_aligned = True
    for name in names[1:]:
        current_time = datasets[name]['time'].values
        if not np.array_equal(reference_time, current_time):
            logging.error(f"Time mismatch found between '{reference_name}' and '{name}'.")
            logging.error(f"'{reference_name}' length: {len(reference_time)}, '{name}' length: {len(current_time)}")
            all_aligned = False
            break
            
    if all_aligned:
        logging.info("✅ All datasets are perfectly aligned in time.")
    return all_aligned

def check_spatial_grid_alignment(datasets: dict) -> bool:
    """
    Checks if all provided datasets have exactly the same latitude and longitude grids.
    """
    logging.info("Running Spatial Grid Alignment Check...")
    
    names = list(datasets.keys())
    reference_name = names[0]
    
    ref_lat = datasets[reference_name]['lat'].values
    ref_lon = datasets[reference_name]['lon'].values

    all_aligned = True
    for name in names[1:]:
        curr_lat = datasets[name]['lat'].values
        curr_lon = datasets[name]['lon'].values
        
        # Using allclose to account for minor floating point differences in netCDF files
        if not np.allclose(ref_lat, curr_lat, atol=1e-5):
            logging.error(f"Latitude mismatch between '{reference_name}' and '{name}'.")
            all_aligned = False
            
        if not np.allclose(ref_lon, curr_lon, atol=1e-5):
            logging.error(f"Longitude mismatch between '{reference_name}' and '{name}'.")
            all_aligned = False

    if all_aligned:
        logging.info("✅ All spatial grids (lat/lon) match perfectly.")
    return all_aligned

def check_land_mask_consistency(datasets: dict, target_var: str = None) -> bool:
    """
    Checks if the NaN mask (representing ocean/outside boundaries) is consistent across datasets.
    """
    logging.info("Running Spatial Land Mask Consistency Check...")
    
    names = list(datasets.keys())
    reference_name = names[0]
    
    # We take the first timestep to compare the spatial mask
    ref_mask = np.isnan(datasets[reference_name].isel(time=0).to_array().values[0])

    all_consistent = True
    for name in names[1:]:
        curr_mask = np.isnan(datasets[name].isel(time=0).to_array().values[0])
        
        if not np.array_equal(ref_mask, curr_mask):
            mismatch_count = np.sum(ref_mask != curr_mask)
            logging.warning(f"Mask inconsistency between '{reference_name}' and '{name}'. "
                            f"{mismatch_count} pixels have mismatched NaN boundaries.")
            all_consistent = False

    if all_consistent:
        logging.info("✅ All land masks and missing value boundaries are consistent.")
    else:
        logging.warning("⚠️ Warning: Land masks differ. Consider creating a unified India land mask and applying it to all datasets.")
        
    return all_consistent

def run_all_qc_checks(datasets: dict):
    """
    Runs the complete suite of national-level quality control checks.
    """
    logging.info("========================================")
    logging.info("STARTING AUTOMATED DATA QUALITY CHECKS")
    logging.info("========================================")
    
    time_ok = check_time_alignment(datasets)
    space_ok = check_spatial_grid_alignment(datasets)
    mask_ok = check_land_mask_consistency(datasets)
    
    logging.info("========================================")
    if time_ok and space_ok and mask_ok:
        logging.info("🚀 ALL QC CHECKS PASSED. Ready for Modeling.")
    else:
        logging.error("❌ QC CHECKS FAILED. Please resolve alignment issues before training.")

# Example Usage:
if __name__ == "__main__":
    # Create dummy datasets for testing the script
    dates = pd.date_range("2000-01-01", "2000-12-31", freq="M")
    lats = np.linspace(8.0, 37.0, 50)
    lons = np.linspace(68.0, 97.0, 50)
    
    # Simulated Rainfall Dataset
    ds_rain = xr.Dataset(
        data_vars=dict(rainfall=(["time", "lat", "lon"], np.random.rand(len(dates), len(lats), len(lons)))),
        coords=dict(time=dates, lat=lats, lon=lons)
    )
    
    # Simulated Temperature Dataset
    ds_temp = xr.Dataset(
        data_vars=dict(tmean=(["time", "lat", "lon"], np.random.rand(len(dates), len(lats), len(lons)))),
        coords=dict(time=dates, lat=lats, lon=lons)
    )
    
    # Run the checks
    run_all_qc_checks({'Rainfall': ds_rain, 'Temperature': ds_temp})
