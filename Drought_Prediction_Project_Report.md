# Project Report: Machine Learning-Based Drought Prediction using Standardized Precipitation Index (SPI) for Eastern Uttar Pradesh

---

## 1. Executive Summary & Project Identification
This project implements an end-to-end Machine Learning pipeline designed to predict the **Standardized Precipitation Index (SPI)**—a standard indicator of meteorological drought—for the **Eastern Uttar Pradesh** region during the critical monsoon season (**JJAS**: June, July, August, September).

Designed to replicate national-grade geophysical research workflows, the pipeline processes raw grid-based NetCDF meteorological data, applies automated Quality Control (QC) checks, trains five distinct machine learning models (including Deep Learning architectures), evaluates them using scientific metrics, generates research-publication figures, and hosts an interactive Streamlit dashboard for real-time inference and long-term future forecasting.

### Key Project Parameters
*   **Target Region:** Eastern Uttar Pradesh (Bounding Box: Latitude $23.5^\circ\text{N} - 28.5^\circ\text{N}$, Longitude $81.0^\circ\text{E} - 84.5^\circ\text{E}$).
*   **Predictor Variables:** Historical Rainfall, Maximum Temperature, and Soil Moisture.
*   **Target Index:** 4-Week Rolling Standardized Precipitation Index (SPI-4) representing drought and wet cycles.
*   **Analysis Interval:** Weekly aggregated steps during the monsoon months (June to September) from 1981 to 2023.
*   **Chronological Split:** Train: 1981–2012 | Test: 2013–2023. This split prevents temporal data leakage.
*   **Winning Model:** Multilayer Perceptron (MLP) with an $R^2$ of **0.911** (2013-2017) and **0.782** (2018-2023).

---

## 2. End-to-End System Architecture

The workflow progresses from raw geospatial files to interactive dashboard outputs. The data and training flow is illustrated below:

```mermaid
graph TD
    A[Raw NetCDF Data: Rainfall, MaxT, Soil Moisture] --> B[data_loader.py: Lazy Loading via Dask]
    B --> C[data_qc_checks.py: Time, Spatial & Mask Alignment]
    C -->|QC Passed| D[01_data_preprocessing.py: Spatial Crop & Filter JJAS]
    D --> E[01_data_preprocessing.py: Resample to Weekly]
    E --> F[01_data_preprocessing.py: Climatology Imputation & SPI-4 Calculation]
    F -->|processed_features.csv| G[02_model_training.py: Sequence Builder Lag = 4]
    G --> H[Chronological Split: Train up to 2012 | Test 2013-2023]
    H --> I[MinMax Scaling & Save Scalers]
    I --> J[Train 5 Models: SVR, RF, XGB, MLP, LSTM]
    J --> K[Evaluate & Save Metrics CSV]
    J --> L[03_evaluation_figures.py: Generate Scientific Charts]
    K & L --> M[03_streamlit_app.py: Run Interactive Dashboard]
    J --> N[04_future_forecast.py: Autoregressive Forecast to 2030]
    N --> M
```

---

## 3. Data Loading & Quality Control Pipeline

### Data Loader (`data_loader.py`)
To work with high-resolution national-scale geophysical datasets without hitting physical memory limits, the system utilizes a **lazy-loading pipeline** implemented with `xarray` and `Dask`.
*   **Lazy Loading:** Rather than reading entire NetCDF grids into RAM immediately, `load_lazy_dataset` opens datasets lazily, splitting coordinates into spatial chunks:
    ```python
    chunk_dict = {'time': -1, 'lat': 100, 'lon': 100}
    ```
*   **Spatial Masking:** It clips the national-scale grid to the exact borders of India early using an ESRI shapefile and `rioxarray`'s clipping API with `all_touched=True` to preserve border grid cells.

### Quality Control Checks (`data_qc_checks.py`)
Before passing the data to the modeling step, the system executes three automated quality control (QC) checks:
1.  **Time-Alignment Check:** Verifies that all three source variables have matching temporal coordinate grids (no missing months or years).
2.  **Spatial Grid Alignment Check:** Checks that latitude and longitude grid points align perfectly across all variables using float tolerances (`np.allclose(atol=1e-5)`).
3.  **Land Mask Consistency Check:** Compares spatial missing value (NaN) boundaries (ocean vs. land grids) across files to prevent computational inconsistencies in multi-variable equations.

---

## 4. Preprocessing & Feature Extraction (`01_data_preprocessing.py`)

The preprocessing script prepares a clean time-series ready for machine learning:

1.  **Spatial Cropping:** The NetCDF coordinates are sorted and sliced to crop the Eastern Uttar Pradesh bounding box ($23.5^\circ\text{N} - 28.5^\circ\text{N}$, $81.0^\circ\text{E} - 84.5^\circ\text{E}$).
2.  **Monsoon Filtering (JJAS):** Slices the time dimension to keep only the months of June (6), July (7), August (8), and September (9).
3.  **Spatial Mean Aggregation:** Computes the spatial average across all grid points in the Eastern UP region to reduce spatial grids into a single regional daily series.
4.  **Weekly Resampling:**
    *   **Rainfall:** Aggregated using a weekly sum (`resample('W').sum()`).
    *   **Max Temperature:** Aggregated using a weekly mean (`resample('W').mean()`).
    *   **Soil Moisture:** Aggregated using a weekly mean (`resample('W').mean()`).
5.  **Climatology Imputation:** Soil moisture observations are missing in the raw dataset for the years 2018–2023. The script calculates a **weekly climatology** (the historical average for each calendar week index across 1981–2017) and imputes the missing values.
6.  **SPI-4 Calculation:**
    The Standardized Precipitation Index (SPI) represents the precipitation anomaly in terms of standard deviation. For a 4-week scale (SPI-4):
    *   Compute the 4-week rolling mean of weekly rainfall:
        $$R_{roll, t} = \frac{1}{4} \sum_{i=0}^{3} R_{t-i}$$
    *   Normalize this rolling series:
        $$SPI_t = \frac{R_{roll, t} - \mu_{roll}}{\sigma_{roll}}$$
        where $\mu_{roll}$ is the historical average of the rolling mean, and $\sigma_{roll}$ is its standard deviation.

The final dataset is saved as `data/processed_features.csv`.

---

## 5. Model Architectures & Training Workflow (`02_model_training.py`)

### Input-Output Sequence Builder
Droughts are slow-onset, multi-week phenomena. To capture temporal dependency, the pipeline converts the data into sequences. It uses a **sequence length of 4 weeks** ($L=4$) of historical variables to predict the current week's SPI:

$$\mathbf{X}_t = \begin{bmatrix}
R_{t-4} & T_{t-4} & SM_{t-4} \\
R_{t-3} & T_{t-3} & SM_{t-3} \\
R_{t-2} & T_{t-2} & SM_{t-2} \\
R_{t-1} & T_{t-1} & SM_{t-1}
\end{bmatrix} \quad \longrightarrow \quad y_t = SPI_t$$

### Chronological Train-Test Split
Rather than random splits (which leak temporal patterns), the dataset is split chronologically:
*   **Training Set:** 1981 to 2012 (32 years of historical data)
*   **Testing Set:** 2013 to 2023 (11 years of unseen validation data)

### Data Normalization
Features are scaled to a $[0, 1]$ range using `MinMaxScaler`. To prevent data leakage, the scaler is fit **only on the training set** and applied to the test set:
*   `models/scaler_X.pkl`: Normalizes inputs (Rainfall, Max Temperature, Soil Moisture).
*   `models/scaler_y.pkl`: Normalizes target (SPI).

### Model Implementations
1.  **Support Vector Regression (SVR):**
    *   **Configuration:** Radial Basis Function (RBF) kernel, regularization parameter $C=1.0$, epsilon tube $\epsilon=0.1$.
    *   **Purpose:** Captures non-linear relationships by projecting features into higher-dimensional spaces.
2.  **Random Forest Regressor (RF):**
    *   **Configuration:** 100 decision trees, maximum tree depth limited to 5 to prevent overfitting.
    *   **Purpose:** An ensemble bagging method that averages predictions from independent decision trees.
3.  **XGBoost Regressor (XGB):**
    *   **Configuration:** 100 trees, maximum tree depth of 3, learning rate of 0.1.
    *   **Purpose:** Gradient boosted trees optimized to minimize residual errors sequentially.
4.  **Multilayer Perceptron (MLP):**
    *   **Architecture:**
        *   Input layer: 12 nodes (flattened $4 \text{ weeks} \times 3 \text{ variables}$).
        *   Dense Layer 1: 32 units (activation: ReLU).
        *   Dropout Layer: rate of 0.2 (prevents co-adaptation of weights).
        *   Dense Layer 2: 16 units (activation: ReLU).
        *   Output Layer: 1 unit (linear activation).
    *   **Optimization:** Adam optimizer, Mean Squared Error (MSE) loss function, and `EarlyStopping` (patience of 15 epochs monitoring validation loss).
5.  **Long Short-Term Memory (LSTM):**
    *   **Architecture:**
        *   LSTM layer: 32 memory units (processes temporal sequences natively).
        *   Dropout Layer: rate of 0.2.
        *   Dense output layer: 1 unit.
    *   **Optimization:** Adam optimizer, MSE loss, and `EarlyStopping`.

---

## 6. Evaluation Metrics & Performance Analysis

The models were evaluated on the test set across two separate intervals: the primary test period (**2013-2017**) and the extended test period containing imputed data (**2018-2023**).

### Performance Metrics Definitions
*   **Coefficient of Determination ($R^2$):** Measures the proportion of variance explained by the model:
    $$R^2 = 1 - \frac{\sum (y_{true} - y_{pred})^2}{\sum (y_{true} - \bar{y}_{true})^2}$$
*   **Root Mean Squared Error (RMSE):** Penalizes larger prediction errors:
    $$RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_{true, i} - y_{pred, i})^2}$$
*   **Mean Absolute Error (MAE):** Represents the average magnitude of absolute errors:
    $$MAE = \frac{1}{N} \sum_{i=1}^{N} |y_{true, i} - y_{pred, i}|$$
*   **Pearson Correlation Coefficient ($r$):** Measures the linear strength of the relationship:
    $$r = \frac{\sum(y_{true} - \bar{y}_{true})(y_{pred} - \bar{y}_{pred})}{\sqrt{\sum(y_{true} - \bar{y}_{true})^2 \sum(y_{pred} - \bar{y}_{pred})^2}}$$

### Metric Comparison Tables

#### Test Period 1: 2013–2017 (Standard Verification)
| Model | $R^2$ Score | RMSE | MAE | Correlation ($r$) |
| :--- | :---: | :---: | :---: | :---: |
| **MLP (Winner)** | **0.911** | **0.275** | **0.219** | **0.957** |
| SVR | 0.873 | 0.329 | 0.274 | 0.956 |
| XGBoost | 0.872 | 0.331 | 0.269 | 0.940 |
| Random Forest | 0.856 | 0.351 | 0.282 | 0.935 |
| LSTM | -0.150 | 0.992 | 0.780 | 0.480 |

#### Test Period 2: 2018–2023 (Extended Verification with Climatology Imputation)
| Model | $R^2$ Score | RMSE | MAE | Correlation ($r$) |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost (Winner)**| **0.791** | **0.415** | **0.307** | **0.890** |
| MLP | 0.782 | 0.425 | **0.302** | 0.886 |
| SVR | 0.764 | 0.441 | 0.371 | 0.882 |
| Random Forest | 0.737 | 0.466 | 0.357 | 0.859 |
| LSTM | -0.071 | 0.941 | 0.704 | 0.340 |

### Key Scientific Insights
1.  **MLP Dominance:** The MLP model performed best in the primary test period ($R^2 = 0.911$), outperforming shallow ensemble architectures like Random Forest and XGBoost. It captured non-linear interactions between temperature, soil moisture, and rainfall anomalies without overfitting.
2.  **LSTM Performance Limitations:** The LSTM model struggled significantly, returning negative $R^2$ values ($-0.150$ and $-0.071$). 
    *   *Why?* LSTMs require long, continuous sequential histories to optimize their hidden states. 
    *   *The Issue:* Because our data is restricted to the monsoon season (JJAS), the temporal record is fragmented (e.g., jumping from September of one year to June of the next). This makes it difficult for the LSTM to learn long-term temporal dependencies, leading to poorer performance on a small dataset (138 training samples).
3.  **Generalization on Imputed Data:** During 2018–2023, where soil moisture values were imputed using historical climatology, the models maintained solid performance ($R^2 \ge 0.737$). This shows the robustness of the features and the effectiveness of the imputation strategy.

---

## 7. Analysis of Graphical Outputs (`outputs/figures/`)

The pipeline generates several figures for analysis and reporting:

### 1. Study Area Map (`Fig1_Study_Area.png`)
*   **Description:** Displays the boundaries of India with Eastern Uttar Pradesh shaded in red.
*   **Purpose:** Confirms the geographical scope ($23.5^\circ\text{N} - 28.5^\circ\text{N}$, $81.0^\circ\text{E} - 84.5^\circ\text{E}$) used for grid-cropping.

### 2. Variable Distribution (`Fig3_Boxplots.png`)
*   **Description:** Side-by-side boxplots showing the distributions of Rainfall, Max Temperature, and Soil Moisture.
*   **Purpose:** Helps identify outliers and highlights differences in measurement scales (e.g., rainfall in mm vs. temperature in $^\circ\text{C}$).

### 3. Model Predictions vs. Actuals (`{Model}_pred_actual.png`)
*   **Description:** Line graphs comparing the yearly mean predicted SPI (red dashed line) with the actual observed SPI (blue solid line). Includes annotated values for each point.
*   **Purpose:** Provides a visual check of how well the model tracks annual wet and dry cycles.

### 4. Residual Distributions (`{Model}_residuals.png`)
*   **Description:** Histograms displaying the distribution of prediction errors ($y_{true} - y_{pred}$).
*   **Purpose:** Helps verify if the residuals are normally distributed around zero, indicating an unbiased model.

### 5. Taylor Diagrams (`Fig11a...` and `Fig11b...`)
*   **Description:** A polar coordinate plot showing three metrics simultaneously: Correlation ($r$ on the radial angle), Standard Deviation (radial distance), and Root Mean Squared Difference (RMSD as concentric contours).
*   **Purpose:** Provides a comprehensive visual comparison of model performance. Models closer to the 'Reference' point on the x-axis are more accurate.

---

## 8. Recursive Multi-Year Forecasting to 2030 (`04_future_forecast.py`)

To forecast future drought trends up to the year 2030, we use an **autoregressive modeling strategy** trained on all available historical data (1981–2023).

### Methodology
1.  **Yearly Aggregation:** Computes yearly mean SPI, minimum and maximum SPI values, standard deviations, and cumulative rainfall totals.
2.  **Autoregressive Features:** Builds input features using a **3-year lag** ($LAG\_YEARS=3$). To predict the SPI of year $Y$, the model uses the stats from years $Y-1$, $Y-2$, and $Y-3$ (24 features total).
3.  **Ensemble Predictor:** An ensemble is created by averaging predictions from four model architectures (Random Forest, Gradient Boosting, MLP, and SVR).
4.  **Recursive Execution:**
    *   To predict the SPI for 2024, the model uses observed stats from 2021-2023.
    *   To predict 2025, the model uses the predicted 2024 SPI, while other meteorological features are gradually decayed toward their historical averages using an alpha decay factor ($\alpha=0.7$):
        $$Feature_{synth, t} = 0.7 \times Feature_{synth, t-1} + 0.3 \times Mean_{historical}$$
    *   This process is repeated recursively up to 2030.

### Year-by-Year Ensemble Predictions (2024-2030)
*   **Average Predicted Forecast SPI ($\bar{X}_{forecast}$):** $-0.057$ (indicating overall near-normal to slightly dry conditions).
*   **Drought Threshold Reference:** Moderate Drought ($<-1.0$), Severe Drought ($<-1.5$).

| Year | Ensemble SPI | Predicted Condition |
| :---: | :---: | :---: |
| **2024** | $-0.034$ | 💧 Near Normal / Wet |
| **2025** | $-0.052$ | 💧 Near Normal / Wet |
| **2026** | $-0.124$ | 🌤️ Mildly Dry |
| **2027** | $-0.092$ | 🌤️ Mildly Dry |
| **2028** | $-0.043$ | 💧 Near Normal / Wet |
| **2029** | $-0.051$ | 💧 Near Normal / Wet |
| **2030** | $-0.051$ | 💧 Near Normal / Wet |

*Observation: The model projects a transition toward slightly drier conditions around 2026-2027, followed by a return to normal levels by 2028-2030. No severe or extreme regional droughts are projected.*

### Percentage Deviation Analysis
The Streamlit app allows comparing the forecasted values ($X$) against two baselines:
*   **Baseline 1 (Avg Forecasted SPI = -0.057):** How much a specific year deviates from the expected 2024-2030 average.
*   **Baseline 2 (Mild Drought Threshold = -0.100):** How close a year is to the dry threshold.

$$\text{Traditional Percentage Deviation} = \frac{X - Baseline}{|Baseline|} \times 100$$
$$\text{Standard Deviation Shift} = (X - Baseline) \times 100$$

---

## 9. Streamlit Dashboard Architecture (`03_streamlit_app.py`)

The user interface is organized into **nine interactive tabs** to help explore data, models, and predictions:

```
├── [Overview & Data]      --> Metadata metrics, dataset sizing, and raw data tables.
├── [Spatial Map]          --> Visualizes Eastern UP boundary coordinates.
├── [Model Comparison]     --> Displays R2, RMSE, MAE, and Taylor Diagrams side-by-side.
├── [Prediction Plots]     --> Compares actual vs. predicted values for each model (2013-2023).
├── [Model Interpretation] --> Features a bar chart showing feature importances.
├── [Decision Summary]     --> Recommends the MLP model and displays its residual plots.
├── [Live Predictor]       --> Predicts the SPI and drought category for the next week based on user input.
├── [Future Forecast]      --> Recursively projects yearly SPI trends up to 2030.
└── [Forecast Percentage]  --> Plots percentage deviations against average and mild baselines.
```

---

## 10. Conclusion & Recommendations

### Project Summary
*   **shallow ML Models vs. Deep Learning:** Shallow models (MLP, SVR, and XGBoost) outperformed LSTMs on this dataset. This is because the seasonal data structure (JJAS) creates gaps in the time-series that affect LSTM convergence.
*   **Predictive Performance:** The MLP model proved highly accurate ($R^2 = 0.911$), making it a reliable option for agricultural planning.
*   **Future Trends:** Autoregressive forecasting projects near-normal monsoon patterns up to 2030, with a minor dry spell expected around 2026-2027.

### Recommendations for Deployment
1.  **Operational Monitoring:** Integrate the live Streamlit predictor with automated weekly weather feeds from agencies like the India Meteorological Department (IMD) for real-time monitoring.
2.  **Mitigation Planning:** Use the forecast to guide agricultural planning and water resource allocation ahead of dry spells.
3.  **Model Expansion:** Consider extending the model to calculate SPI-12 (long-term hydrological drought) to assist in groundwater management.
