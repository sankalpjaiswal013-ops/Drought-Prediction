import sys

print("Checking for GRIB2 Python libraries...")

try:
    import cfgrib
    print("✅ 'cfgrib' is installed and available!")
except ImportError:
    print("❌ 'cfgrib' is NOT installed.")

try:
    import pygrib
    print("✅ 'pygrib' is installed and available!")
except ImportError:
    print("❌ 'pygrib' is NOT installed.")

# Print python version and environment info
print(f"\nPython Version: {sys.version}")
print(f"Python Executable: {sys.executable}")
