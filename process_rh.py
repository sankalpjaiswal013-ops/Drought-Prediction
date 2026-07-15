import os
import tarfile
import tempfile
import shutil
import rasterio
import pandas as pd
import numpy as np
from datetime import datetime

# Paths
tar_path = "raw_data/RH/RH_2m_hrly_19790101_20201231.tar.gz"
output_csv = "data/processed_rh.csv"

# Bounding box for Eastern Uttar Pradesh
LAT_MIN, LAT_MAX = 23.5, 28.5
LON_MIN, LON_MAX = 81.0, 84.5

def process_rh_tarball():
    if not os.path.exists(tar_path):
        print(f"❌ Could not find Relative Humidity tarball at: {tar_path}")
        return

    print("🚀 Starting Highly Optimized GRIB2 Relative Humidity Preprocessing Pipeline...")
    print(f"Archive Size: {os.path.getsize(tar_path) / (1024**3):.2f} GB")
    
    # Create temporary directory for extraction
    temp_dir = tempfile.mkdtemp(prefix="rh_process_")
    print(f"📁 Created temporary extraction folder: {temp_dir}")
    
    data_records = []
    
    try:
        # Open the tarball as a stream for speed
        with tarfile.open(tar_path, "r|gz") as tar:
            count = 0
            processed_count = 0
            
            for member in tar:
                # We only care about .grb2 files
                if not member.name.endswith(".grb2"):
                    continue
                
                count += 1
                base_name = os.path.basename(member.name)
                
                # Filename example: RH-2m_1984032203_ncum_imdaa_reanl_2df.grb2
                try:
                    parts = base_name.split("_")
                    date_str = parts[1] # "1984032203"
                    year = int(date_str[0:4])
                    month = int(date_str[4:6])
                    day = int(date_str[6:8])
                    hour = int(date_str[8:10])
                except Exception:
                    # Skip files that don't match our pattern
                    continue
                
                # Strict spatial and time filtering:
                # 1. Years: 1981 to 2020 (RH data ends at 2020)
                # 2. Months: JJAS (June = 6, July = 7, August = 8, September = 9)
                if year < 1981 or year > 2020 or month not in [6, 7, 8, 9]:
                    continue
                
                processed_count += 1
                if processed_count % 500 == 1:
                    print(f"⌛ Processed {processed_count} GRIB2 records. Currently on date: {year}-{month:02d}-{day:02d}...")
                
                # Extract only this single GRIB2 file
                tar.extract(member, path=temp_dir)
                extracted_file_path = os.path.join(temp_dir, member.name)
                
                try:
                    # Read using rasterio (GDAL GRIB Driver)
                    with rasterio.open(extracted_file_path) as src:
                        # Get pixel indexes for our bounding box
                        row_top, col_left = src.index(LON_MIN, LAT_MAX)
                        row_bottom, col_right = src.index(LON_MAX, LAT_MIN)
                        
                        r_min = min(row_top, row_bottom)
                        r_max = max(row_top, row_bottom)
                        c_min = min(col_left, col_right)
                        c_max = max(col_left, col_right)
                        
                        # Read the cropped window (Band 1)
                        window = rasterio.windows.Window(c_min, r_min, c_max - c_min, r_max - r_min)
                        data = src.read(1, window=window)
                        
                        # Calculate mean relative humidity in the cropped area
                        mean_rh = np.nanmean(data)
                        
                        # Store record
                        timestamp = datetime(year, month, day, hour)
                        data_records.append({"time": timestamp, "RH": mean_rh})
                        
                except Exception as e:
                    print(f"\n⚠️ Error processing {base_name}: {e}")
                finally:
                    # Clean up file immediately to save disk
                    if os.path.exists(extracted_file_path):
                        os.remove(extracted_file_path)
                        
    except Exception as e:
        print(f"\n❌ Critical error reading tarball: {e}")
        
    finally:
        # Clean up the temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"\n🧹 Cleaned up temporary directory: {temp_dir}")
            
    if data_records:
        print("\n💾 Saving processed Relative Humidity records...")
        full_df = pd.DataFrame(data_records).sort_values("time").set_index("time")
        
        # Save daily averages to make the CSV extremely compact
        print("Resampling hourly RH data to Daily Mean to save space...")
        daily_df = full_df.resample('D').mean()
        
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        daily_df.to_csv(output_csv)
        print(f"🎉 Success! Processed RH data saved to {output_csv}")
    else:
        print("❌ No data was processed. Please check if your GRIB2 files match the naming convention.")

if __name__ == "__main__":
    process_rh_tarball()
