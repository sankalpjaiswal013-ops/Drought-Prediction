# Uttar Pradesh Drought Prediction using Machine Learning

## Project Overview
This project is an end-to-end Machine Learning pipeline designed to predict the **Standardized Precipitation Index (SPI)** (drought conditions) for the **Eastern Uttar Pradesh** region during the monsoon season (JJAS: June, July, August, September).

The project was modeled after national-grade geophysical research workflows and processes raw NetCDF meteorological data, trains multiple predictive models, and visualizes the results through an interactive Streamlit dashboard.

## Dataset Details
- **Inputs:** Historical Rainfall, Maximum Temperature, and Soil Moisture.
- **Geographic Scope:** Strictly cropped to the Eastern Uttar Pradesh bounding box (Lat: 23.5°N - 28.5°N, Lon: 81.0°E - 84.5°E).
- **Target Variable:** 4-Week Rolling Standardized Precipitation Index (SPI).
- **Train/Test Split:** Chronological split (Training: up to 2012, Testing: 2013-2017) to prevent data leakage and ensure real-world validity.

## Machine Learning Models Used
Five state-of-the-art models were trained and rigorously compared:
1. **Multilayer Perceptron (MLP)** *(Best Performing)*
2. **Support Vector Regression (SVR)**
3. **Random Forest Regressor**
4. **XGBoost Regressor**
5. **Long Short-Term Memory (LSTM)**

## How to Run the Project

### 1. Data Preprocessing
Extracts data from raw `.nc` files, crops to Eastern UP, and aggregates weekly:
```bash
python 01_data_preprocessing.py
```

### 2. Model Training
Trains all 5 models and evaluates their performance on the unseen 2013-2017 test set:
```bash
python 02_model_training.py
```

### 3. Generate Scientific Figures
Generates the Spatial Map, Boxplots, and the Taylor Diagram for research paper inclusion:
```bash
python 03_evaluation_figures.py
```

### 4. Launch the Dashboard
Opens the interactive web application to visualize metrics, predictions, and perform live manual forecasting:
```bash
# Standard command:
streamlit run 03_streamlit_app.py

# Alternative (if streamlit.exe is blocked by Windows Application Control policy):
python -m streamlit run 03_streamlit_app.py
```

## Key Findings
The **MLP model** significantly outperformed the other architectures, successfully mapping the non-linear relationship of the meteorological variables to the SPI index. Deep learning models like LSTM struggled to converge due to the highly condensed nature of the weekly aggregated dataset (limited sequence length), highlighting that shallower architectures (MLP, SVR) are far more appropriate for datasets of this scale.
