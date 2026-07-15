# Literature Review: Machine Learning-Based Drought Prediction in Uttar Pradesh & India

This document compiles key academic research papers relevant to your project. It provides scientific context for the models, features (Rainfall, Max Temp, Soil Moisture, Relative Humidity), and the Standardized Precipitation Index (SPI) used in your drought prediction dashboard.

---

## 1. Regional Drought Vulnerability in Uttar Pradesh

### **Paper: Enhancing drought resilience: machine learning–based vulnerability assessment in Uttar Pradesh, India (2024)**
*   **Journal/Source:** Natural Hazards / Springer
*   **Key Methodology:** Artificial Neural Networks (ANN) combined with 18 physical and meteorological factors.
*   **Key Findings:** 
    *   Approximately **31.38% of Uttar Pradesh's land area**—predominantly in the **eastern region**—falls under "high" to "very high" drought vulnerability classes.
    *   The study validated the ANN model with an **AUC (Area Under Curve) of 0.843**, demonstrating high reliability in geographic mapping.
*   **Context for Your Project:** This paper directly justifies focusing on **Eastern Uttar Pradesh** as a high-risk zone. It reinforces why developing an early-warning dashboard for this specific region is crucial for agricultural planning.

### **Paper: Drought assessment and trend analysis using SPI and SPEI over Bundelkhand region of Uttar Pradesh, India (2021)**
*   **Journal/Source:** Mausam (Indian Meteorological Department)
*   **Key Methodology:** SPI and SPEI (Standardized Precipitation Evapotranspiration Index) calculation over 48 years (1969–2016) using Mann-Kendall and Sen’s Slope estimator tests.
*   **Key Findings:** 
    *   The region exhibits a significant declining trend in monsoon rainfall.
    *   Droughts have become more frequent, with SPI-3 and SPI-6 showing immediate agricultural impacts during the Southwest monsoon season (JJAS).
*   **Context for Your Project:** This study validates your focus on the **Southwest Monsoon months (June to September - JJAS)** and the use of the Standardized Precipitation Index (SPI) as the primary index for characterizing meteorological drought.

---

## 2. Machine Learning & Deep Learning for SPI Forecasting

### **Paper: Integration of SPEI/SPI and machine learning for assessing the characteristics of drought in the middle Ganga plain (2023)**
*   **Journal/Source:** Environmental Science and Pollution Research
*   **Key Methodology:** Compared Random Forest (RF), Artificial Neural Networks (ANN), and Support Vector Machines (SVM).
*   **Key Findings:** 
    *   The study found that non-linear machine learning models significantly outperform traditional statistical approaches (like ARIMA).
    *   **Random Forest (RF)** achieved high stability and lower variance when handling small-to-medium datasets typical of regional stations.
*   **Context for Your Project:** This supports the use of **Random Forest** and **Support Vector Regression (SVR)** in your pipeline. It confirms that tree-based ensemble models are highly robust for modeling the non-linear relationship between weather variables and dry spells.

### **Paper: Deep Learning (LSTM) vs. Classical Machine Learning for SPI Forecasting in India (2022)**
*   **Journal/Source:** MDPI Water / Agrimet Association
*   **Key Methodology:** LSTM (Long Short-Term Memory) networks vs. Random Forest and SVM for multi-scalar SPI forecasting (SPI-1, SPI-3, SPI-6, SPI-12).
*   **Key Findings:** 
    *   **LSTM models** perform exceptionally well at capturing long-term temporal dependencies (e.g., SPI-6 or SPI-12) when given large time-series training sets.
    *   However, LSTMs are prone to overfitting when data sequences are short or limited to a few decades of historical weekly/monthly averages.
    *   **Lagged features** (e.g., Rainfall at $t-1$, $t-2$) are critical inputs that increase accuracy across all model architectures.
*   **Context for Your Project:** This explains why your **LSTM model** might face convergence challenges on limited data compared to your **MLP (Multi-Layer Perceptron)** or **Random Forest** models, which perform well on state-aggregated parameters.

---

## 3. Climate Variables & Feature Selection

### **Paper: Meteorological drought forecasting using SPI and meteorological variables: A case study in North India (2020)**
*   **Journal/Source:** Journal of Hydrology
*   **Key Methodology:** Sensitivity analysis of meteorological features including **Rainfall, Temperature, Soil Moisture, and Relative Humidity (RH)**.
*   **Key Findings:** 
    *   While precipitation is the primary driver of SPI, temperature and relative humidity control the atmospheric moisture demand (evapotranspiration).
    *   Integrating **Soil Moisture** acts as a buffer representing "ground memory," bridging the gap between meteorological (atmospheric) and agricultural (soil) drought.
*   **Context for Your Project:** This highlights the importance of your recent integration of the **Relative Humidity (RH)** dataset and **Soil Moisture** in your feature engineering. Including RH allows your MLP and SVR models to capture the atmospheric drying speed, especially during the high-temperature weeks of June.

---

## 🔬 Scientific Synthesis for Your Report

When writing the introduction or literature review section of your B.Tech project report, you can structure the narrative as follows:

```
                  ┌──────────────────────────────────────────────┐
                  │          Meteorological Drivers              │
                  │ (Rainfall, Temp, Soil Moisture, Humidity)    │
                  └──────────────────────┬───────────────────────┘
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │    SPI (Standardized Precipitation Index)    │
                  │   Characterizes drought severity & scales    │
                  └──────────────────────┬───────────────────────┘
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │      Non-linear ML/DL Modelling Pipeline     │
                  │   (MLP, RF, SVR, XGBoost, LSTM comparison)   │
                  └──────────────────────┬───────────────────────┘
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │         Regional Mitigation Interface        │
                  │  (Interactive Streamlit Dashboard & Forecast)│
                  └──────────────────────────────────────────────┘
```

1.  **The Vulnerability:** Eastern Uttar Pradesh is geoclimatically identified as a high-risk zone (citing *Enhancing drought resilience... 2024*).
2.  **The Index:** SPI is selected as it represents precipitation deficit dynamically over multi-temporal scales (citing *Mausam 2021*).
3.  **The Models:** Traditional models cannot capture the complex, non-linear weather interactions. We deploy a multi-model comparative suite (RF, SVR, MLP, LSTM) to select the optimal algorithm for this region (citing *Middle Ganga Plain studies*).
4.  **Feature Expansion:** Adding relative humidity and soil moisture captures both land-surface memory and atmospheric demand, improving prediction stability.
