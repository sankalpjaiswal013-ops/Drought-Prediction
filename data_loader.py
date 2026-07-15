import xarray as xr
import rioxarray
import geopandas as gpd
import logging
import dask
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_lazy_dataset(file_path_or_pattern: str, chunk_dict: dict = None) -> xr.Dataset:
    """
    Loads NetCDF data lazily using Dask to prevent MemoryErrors on national-scale data.
    
    Args:
        file_path_or_pattern: Path to a single file or a glob pattern (e.g., 'data/raw/rain_*.nc')
        chunk_dict: Dictionary specifying chunk sizes, e.g., {'time': -1, 'lat': 50, 'lon': 50}.
                    -1 means no chunking along that dimension.
    """
    if chunk_dict is None:
        # Default chunking strategy: Keep time contiguous, chunk spatially
        chunk_dict = {'time': -1, 'lat': 100, 'lon': 100}
        
    logging.info(f"Loading data from {file_path_or_pattern} with chunks {chunk_dict}...")
    
    # Use open_mfdataset for multiple files, or open_dataset for a single file
    if '*' in file_path_or_pattern or '?' in file_path_or_pattern:
        ds = xr.open_mfdataset(file_path_or_pattern, chunks=chunk_dict, parallel=True)
    else:
        ds = xr.open_dataset(file_path_or_pattern, chunks=chunk_dict)
        
    logging.info(f"Loaded dataset size: {ds.nbytes / (1024**3):.2f} GB (Lazy evaluation)")
    return ds

def apply_india_mask(ds: xr.Dataset, shapefile_path: str) -> xr.Dataset:
    """
    Clips the dataset using the India boundary shapefile. 
    Doing this early ensures consistent land/ocean masking across all variables.
    """
    logging.info(f"Applying India spatial mask using {shapefile_path}...")
    
    if not os.path.exists(shapefile_path):
        raise FileNotFoundError(f"Shapefile not found at {shapefile_path}")

    # Load shapefile
    india_shape = gpd.read_file(shapefile_path)
    
    # Ensure spatial dimensions are recognized by rioxarray
    if 'lon' in ds.dims and 'lat' in ds.dims:
        ds = ds.rio.set_spatial_dims(x="lon", y="lat", inplace=True)
    elif 'longitude' in ds.dims and 'latitude' in ds.dims:
        ds = ds.rio.set_spatial_dims(x="longitude", y="latitude", inplace=True)
    else:
        raise ValueError("Could not find standard lat/lon dimensions in dataset.")
        
    ds = ds.rio.write_crs("epsg:4326", inplace=True)
    
    # Clip the dataset to the shapefile boundary
    # all_touched=True ensures we don't aggressively drop border pixels
    masked_ds = ds.rio.clip(india_shape.geometry.values, india_shape.crs, drop=True, all_touched=True)
    
    logging.info("Mask applied successfully.")
    return masked_ds

def build_data_pipeline(rain_path, temp_path, soil_path, shapefile_path):
    """
    Orchestrates the loading and masking of all required datasets.
    """
    logging.info("--- Starting Dask/xarray Data Pipeline ---")
    
    # 1. Load data lazily
    # Note: Adjust chunks based on your specific grid resolution and RAM
    spatial_chunks = {'time': -1, 'lat': 50, 'lon': 50} 
    
    ds_rain = load_lazy_dataset(rain_path, chunk_dict=spatial_chunks)
    ds_temp = load_lazy_dataset(temp_path, chunk_dict=spatial_chunks)
    ds_soil = load_lazy_dataset(soil_path, chunk_dict=spatial_chunks)
    
    # 2. Apply the India mask early
    ds_rain_masked = apply_india_mask(ds_rain, shapefile_path)
    ds_temp_masked = apply_india_mask(ds_temp, shapefile_path)
    ds_soil_masked = apply_india_mask(ds_soil, shapefile_path)
    
    datasets = {
        'Rainfall': ds_rain_masked,
        'Temperature': ds_temp_masked,
        'Soil Moisture': ds_soil_masked
    }
    
    logging.info("--- Data Pipeline Initialization Complete ---")
    return datasets

# Example usage (Uncomment and adjust paths when ready to run)
if __name__ == "__main__":
    pass
    # SHAPEFILE_PATH = "data/raw/india_shapefile/india_boundary.shp"
    # RAIN_DATA = "data/raw/rainfall/*.nc"
    # TEMP_DATA = "data/raw/temperature/*.nc"
    # SOIL_DATA = "data/raw/soil_moisture/*.nc"
    # 
    # datasets = build_data_pipeline(RAIN_DATA, TEMP_DATA, SOIL_DATA, SHAPEFILE_PATH)
    # 
    # # Now you can pass these directly into the QC checks!
    # from data_qc_checks import run_all_qc_checks
    # run_all_qc_checks(datasets)
