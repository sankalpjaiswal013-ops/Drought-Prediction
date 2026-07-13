import subprocess
import sys
import os

def run_script(script_name):
    print("\n" + "="*80)
    print(f"  RUNNING SCRIPT: {script_name}")
    print("="*80)
    try:
        # Run using the same python interpreter
        result = subprocess.run([sys.executable, script_name], check=True, text=True)
        print(f"✓ {script_name} completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error executing {script_name}: {e}")
        return False

if __name__ == "__main__":
    print("Starting Drought Prediction Master Pipeline...")
    
    # Check if raw data folders exist
    required_dirs = [
        "raw_data/rainfall",
        "raw_data/Maximum Temprature",
        "raw_data/soil",
        "raw_data/east up"
    ]
    missing = [d for d in required_dirs if not os.path.exists(d)]
    if missing:
        print(f"Warning: The following directories are missing: {missing}")
        print("Please ensure your raw NetCDF and shapefiles are in raw_data/ before running.")
        
    steps = [
        "01_data_preprocessing.py",
        "02_model_training.py",
        "03_evaluation_figures.py",
        "04_future_forecast.py"
    ]
    
    success = True
    for step in steps:
        if not run_script(step):
            success = False
            break
            
    if success:
        print("\n" + "="*80)
        print("🎉 MASTER PIPELINE SUCCESSFUL!")
        print("All models are retrained, evaluation figures regenerated, and future forecasts updated.")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("❌ MASTER PIPELINE FAILED.")
        print("Please check the error logs above.")
        print("="*80)
