import os
import tarfile
import tempfile
import shutil
import rasterio

tar_path = "raw_data/RH/RH_2m_hrly_19790101_20201231.tar.gz"

if not os.path.exists(tar_path):
    print(f"❌ Could not find tarball at {tar_path}")
else:
    print(f"📦 Opening {tar_path}...")
    temp_dir = tempfile.mkdtemp(prefix="rh_test_")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            # Find the first .grb2 member
            grb2_member = None
            for m in tar.getmembers():
                if m.name.endswith(".grb2"):
                    grb2_member = m
                    break
            
            if grb2_member is None:
                print("❌ No .grb2 files found in tarball!")
            else:
                print(f"📄 Extracting test member: {grb2_member.name}...")
                tar.extract(grb2_member, path=temp_dir)
                file_path = os.path.join(temp_dir, grb2_member.name)
                
                print(f"🔍 Opening file with rasterio: {file_path}")
                with rasterio.open(file_path) as src:
                    print("✅ Successfully opened GRIB2 file with rasterio!")
                    print(f"Metadata: {src.meta}")
                    print(f"Bounds: {src.bounds}")
                    print(f"Width: {src.width}, Height: {src.height}")
                    # Read the first band
                    data = src.read(1)
                    print(f"Read band 1 successfully. Data shape: {data.shape}")
                    print(f"Sample data value: {data[data.shape[0]//2, data.shape[1]//2]}")
                    
    except Exception as e:
        print(f"❌ Error during test: {e}")
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print("🧹 Cleaned up temp files.")
