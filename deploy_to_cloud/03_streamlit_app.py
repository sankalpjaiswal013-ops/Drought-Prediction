import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Eastern Uttar Pradesh Drought Prediction",
    page_icon="🌦️",
    layout="wide"
)

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed_features.csv", parse_dates=['time'])
    metrics = pd.read_csv("outputs/results/metrics.csv")
    metrics_2 = pd.read_csv("outputs/results/metrics_2018_2023.csv")
    return df, metrics, metrics_2

try:
    df, metrics_df, metrics_df_2 = load_data()
    data_loaded = True
    start_year = df['time'].dt.year.min()
    end_year = df['time'].dt.year.max()
except Exception as e:
    st.error(f"Error loading data. Make sure you have run the preprocessing and training scripts. ({e})")
    data_loaded = False
    start_year = "Unknown"
    end_year = "Unknown"

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.title("🌦️ Drought Prediction Dashboard")
st.sidebar.markdown(f"""
**Project:** Eastern UP SPI Prediction  
**Region:** Eastern Uttar Pradesh (JJAS)  
**Period:** {start_year} - {end_year}  
""")

tab_selection = st.sidebar.radio(
    "Navigation",
    ["Overview & Data", "Spatial Map", "Model Comparison", "Prediction Plots", "Model Interpretation", "Decision Summary", "Live Predictor", "Future Forecast", "Forecast Percentage"]
)

if data_loaded:
    # ---------------------------------------------------------
    # Tab 1: Overview & Data
    # ---------------------------------------------------------
    if tab_selection == "Overview & Data":
        st.title("Dataset Overview")
        st.markdown("""
        This dashboard presents the results of predicting the **Standardized Precipitation Index (SPI)** 
        for the Eastern Uttar Pradesh region using meteorological data (Rainfall, Max Temp, Soil Moisture) 
        during the monsoon season (JJAS).
        """)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Weeks Recorded", len(df))
        col2.metric("Start Year", df['time'].dt.year.min())
        col3.metric("End Year", df['time'].dt.year.max())
        
        st.subheader("Raw Processed Data")
        st.dataframe(df.tail(10), width="stretch")

    # ---------------------------------------------------------
    # Tab 2: Spatial Map
    # ---------------------------------------------------------
    elif tab_selection == "Spatial Map":
        st.title("🗺️ Spatial Map of Study Area")
        st.markdown("""
        To predict droughts accurately, we must strictly define the geographic boundaries. 
        This map highlights the exact coordinates for the **Eastern Uttar Pradesh region** used to aggregate 
        our meteorological NetCDF grid points.
        """)
        try:
            st.image("outputs/figures/Fig1_Study_Area.png", caption="Eastern Uttar Pradesh Study Area", use_column_width=False)
        except:
            st.info("Study area map not found. Run `03_evaluation_figures.py` to generate it.")

    # ---------------------------------------------------------
    # Tab 2: Model Comparison
    # ---------------------------------------------------------
    elif tab_selection == "Model Comparison":
        st.title("Model Comparison Metrics")
        st.markdown("Comparing the performance of SVR, Random Forest, MLP, and LSTM on the 2013-2023 test set.")
        
        # Display Metrics Tables
        st.subheader("Test Set: 2013-2017")
        st.dataframe(metrics_df.style.highlight_max(subset=['R2', 'Correlation'], color='lightgreen')
                                       .highlight_min(subset=['RMSE', 'MAE'], color='lightgreen'),
                     width="stretch")
                     
        st.subheader("Test Set: 2018-2023")
        st.dataframe(metrics_df_2.style.highlight_max(subset=['R2', 'Correlation'], color='lightgreen')
                                       .highlight_min(subset=['RMSE', 'MAE'], color='lightgreen'),
                     width="stretch")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Taylor Diagrams")
            try:
                st.image("outputs/figures/Fig11a_Taylor_Diagram_2013_2017.png", caption="Taylor Diagram (2013-2017)")
                st.image("outputs/figures/Fig11b_Taylor_Diagram_2018_2023.png", caption="Taylor Diagram (2018-2023)")
            except:
                st.info("Taylor diagrams not found.")
                
        with col2:
            st.subheader("Variable Distribution")
            try:
                st.image("outputs/figures/Fig3_Boxplots.png", caption="Boxplots of Input Features")
            except:
                st.info("Boxplots not found.")
                
        # Export button for metrics
        csv_1 = metrics_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download 2013-2017 Metrics as CSV",
            data=csv_1,
            file_name='metrics_2013_2017.csv',
            mime='text/csv',
        )
        
        csv_2 = metrics_df_2.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download 2018-2023 Metrics as CSV",
            data=csv_2,
            file_name='metrics_2018_2023.csv',
            mime='text/csv',
        )

    # ---------------------------------------------------------
    # Tab 3: Prediction Plots
    # ---------------------------------------------------------
    elif tab_selection == "Prediction Plots":
        st.title("Time-Series Predictions (Test Set)")
        st.markdown("Visualizing the Predicted SPI against the Actual SPI for the validation period.")
        
        model_choice = st.selectbox("Select Model to View:", ["MLP", "SVR", "RF", "LSTM", "XGB"])
        
        try:
            st.image(f"outputs/figures/{model_choice}_pred_actual.png", 
                     caption=f"{model_choice} - Actual vs Predicted SPI", 
                     use_column_width=True)
        except:
            st.error(f"Prediction plot for {model_choice} not found.")

    # ---------------------------------------------------------
    # Tab 4: Model Interpretation
    # ---------------------------------------------------------
    elif tab_selection == "Model Interpretation":
        st.title("🧠 Model Interpretation (Feature Importance)")
        st.markdown("""
        To make this a transparent, 'serious' project, we must understand *how* the model makes its decisions. 
        Below is the **Feature Importance** chart derived from the Random Forest model. It proves which meteorological 
        variable has the strongest mathematical impact on predicting a drought in Eastern Uttar Pradesh.
        """)
        
        with st.spinner("Calculating interpretability metrics..."):
            from sklearn.ensemble import RandomForestRegressor
            
            # Train an interpretation model
            X_interp = df[['Rainfall', 'Max_Temp', 'Soil_Moisture']]
            y_interp = df['SPI']
            
            interp_rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
            interp_rf.fit(X_interp, y_interp)
            
            # Extract importances
            importances = interp_rf.feature_importances_
            feature_names = X_interp.columns
            
            # Create a dataframe for plotting
            imp_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance (0 to 1)': importances
            }).sort_values(by='Importance (0 to 1)', ascending=False)
            
            st.bar_chart(imp_df.set_index('Feature'))
            
            st.markdown("""
            ### Interpretation:
            - **Rainfall** is typically the most dominant factor, as SPI is directly derived from precipitation deficits.
            - **Soil Moisture** acts as the crucial memory component of the land, dictating how severe the drought actually is on the ground.
            - **Max Temperature** contributes by driving evaporation, which accelerates drought conditions.
            """)

    # ---------------------------------------------------------
    # Tab 6: Decision Summary
    # ---------------------------------------------------------
    elif tab_selection == "Decision Summary":
        st.title("🏆 Decision Summary")
        
        # Dynamically find the best model from the 2018-2023 metrics
        best_model_row = metrics_df_2.loc[metrics_df_2['R2'].idxmax()]
        best_model_name = best_model_row['Model']
        best_r2 = best_model_row['R2']
        best_rmse = best_model_row['RMSE']
        best_corr = best_model_row['Correlation']
        
        st.markdown(f"### The Winning Model: **{best_model_name}**")
        st.markdown(f"""
        After rigorous chronological validation spanning the extended test period (2018-2023), the **{best_model_name}** emerged as the absolute best algorithm 
        for mapping complex weather variables to drought indices in Eastern Uttar Pradesh.
        """)
        
        # Display Best Model Scores dynamically
        col1, col2, col3 = st.columns(3)
        col1.metric("Winning R² Score", f"{best_r2:.3f}")
        col2.metric("Winning RMSE", f"{best_rmse:.3f}")
        col3.metric("Correlation", f"{best_corr:.3f}")
        
        st.markdown("---")
        st.markdown("""
        ### Residual Analysis
        Examine the residual distribution below to verify that the model does not suffer from systemic bias.
        """)
        
        model_choice = st.selectbox("Select Model for Residual Analysis:", ["MLP", "SVR", "RF", "LSTM", "XGB"])
        
        try:
            st.image(f"outputs/figures/{model_choice}_residuals.png", 
                     caption=f"{model_choice} - Residual Distribution", 
                     use_column_width=False)
        except:
            st.error("Residual plot not found.")
            
        st.markdown("""
        ### Final Recommendation
        - **Deep Learning Challenge:** The LSTM model struggled significantly due to the severely limited sequence length available (138 training samples). Deep recurrent models require vastly larger datasets to converge properly.
        - **Operational Deployment:** The **MLP model** is highly recommended for deployment. It can reliably predict 1-week advance SPI drought categories (Normal, Mildly Dry, Moderate Drought, Severe Drought) using live state-aggregated inputs.
        """)

    # ---------------------------------------------------------
    # Tab 5: Live Predictor
    # ---------------------------------------------------------
    elif tab_selection == "Live Predictor":
        st.title("📡 Live Drought Predictor (State-Aggregated)")
        st.markdown("""
        **How it works:**  
        The model is trained on historical JJAS climate data for Eastern Uttar Pradesh to learn drought patterns. 
        During real-time prediction, the system uses current state-aggregated weather data from live sources 
        as inputs to estimate the SPI and current drought status.
        """)
        
        st.subheader("Input Current Week's Data (UP State Average)")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            live_rain = st.number_input("Rainfall (mm/week)", value=50.0, min_value=0.0, max_value=400.0)
        with col2:
            live_temp = st.number_input("Max Temp (°C)", value=35.0, min_value=20.0, max_value=50.0)
        with col3:
            live_soil = st.number_input("Soil Moisture (kg/m²)", value=300.0, min_value=100.0, max_value=600.0)
            
        if st.button("Predict Next Week's SPI", type="primary"):
            with st.spinner("Analyzing patterns..."):
                # Fast on-the-fly training for demo purposes
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.preprocessing import MinMaxScaler
                
                # Setup dummy sequence simulating recent past + live input
                features = df[['Rainfall', 'Max_Temp', 'Soil_Moisture']].values
                target = df['SPI'].values
                
                # Train model
                scaler_X = MinMaxScaler()
                scaler_y = MinMaxScaler()
                
                X_sc = scaler_X.fit_transform(features)
                y_sc = scaler_y.fit_transform(target.reshape(-1, 1)).ravel()
                
                # We use Random Forest because Linear Regression suffers from 
                # multicollinearity coefficient explosion (causing the 22.86 error)
                demo_rf = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42)
                demo_rf.fit(X_sc, y_sc)
                
                # Predict
                live_input = np.array([[live_rain, live_temp, live_soil]])
                live_input_sc = scaler_X.transform(live_input)
                
                pred_sc = demo_rf.predict(live_input_sc)
                final_spi = scaler_y.inverse_transform(pred_sc.reshape(-1, 1))[0][0]
                
                st.markdown("---")
                st.subheader(f"Predicted SPI: **{final_spi:.2f}**")
                
                # Interpret the SPI
                if final_spi > 0:
                    st.success("💧 Status: Normal / Wet Conditions. No drought expected.")
                elif final_spi > -1.0:
                    st.info("🌤️ Status: Mildly Dry. Keep monitoring.")
                elif final_spi > -1.5:
                    st.warning("⚠️ Status: Moderate Drought Warning!")
                else:
                    st.error("🚨 Status: Severe Drought Alert!")

    # ---------------------------------------------------------
    # Tab 8: Future Forecast (2024-2030)
    # ---------------------------------------------------------
    elif tab_selection == "Future Forecast":
        st.title("🔮 Future Drought Forecast (2024-2030)")
        st.markdown("""
        **Methodology:**  
        An ensemble of 4 ML models (Random Forest, Gradient Boosting, MLP, SVR) was trained on **all** 
        historical yearly mean SPI data (1981-2023) using an autoregressive approach. Past 5 years of 
        SPI and climate statistics are used to recursively forecast each future year.
        """)
        
        try:
            forecast_df = pd.read_csv("outputs/results/forecast_2024_2030.csv")
            
            # Key metrics
            col1, col2, col3 = st.columns(3)
            worst_year = forecast_df.loc[forecast_df['Ensemble_SPI'].idxmin()]
            best_year = forecast_df.loc[forecast_df['Ensemble_SPI'].idxmax()]
            col1.metric("Most Vulnerable Year", int(worst_year['Year']), f"SPI: {worst_year['Ensemble_SPI']:.3f}")
            col2.metric("Safest Year", int(best_year['Year']), f"SPI: {best_year['Ensemble_SPI']:.3f}")
            col3.metric("Avg Forecast SPI", f"{forecast_df['Ensemble_SPI'].mean():.3f}")
            
            st.markdown("---")
            
            # Forecast plot
            st.subheader("Forecast Visualization")
            try:
                st.image("outputs/figures/Fig12_Forecast_2030.png", 
                         caption="Ensemble Drought Forecast to 2030", width=900)
            except:
                st.info("Forecast plot not found. Run `python 04_future_forecast.py` first.")
            
            # Forecast table
            st.subheader("Year-by-Year Forecast")
            display_df = forecast_df.copy()
            display_df['Year'] = display_df['Year'].astype(int)
            display_df['Status'] = display_df['Ensemble_SPI'].apply(
                lambda x: '💧 Normal/Wet' if x > 0 else ('🌤️ Mildly Dry' if x > -1.0 else ('⚠️ Moderate Drought' if x > -1.5 else '🚨 Severe Drought'))
            )
            st.dataframe(display_df.style.format({
                'Ensemble_SPI': '{:.3f}',
                'Random Forest': '{:.3f}',
                'Gradient Boosting': '{:.3f}',
                'MLP': '{:.3f}',
                'SVR': '{:.3f}'
            }), width="stretch")
            
            # Feature importance
            st.subheader("Feature Importance for Forecasting")
            try:
                st.image("outputs/figures/Fig13_Forecast_Feature_Importance.png",
                         caption="Top Autoregressive Features Driving the Forecast")
            except:
                st.info("Feature importance plot not found.")
            
            # Download button
            csv = forecast_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Forecast Data as CSV",
                data=csv,
                file_name='drought_forecast_2024_2030.csv',
                mime='text/csv',
            )
            
            st.markdown("""
            ### ⚠️ Important Disclaimer
            - These forecasts are **statistical projections** based on historical patterns, not physical climate simulations.
            - Accuracy naturally decreases for years further from the training period.
            - The ensemble approach averages 4 different model architectures to reduce individual model bias.
            - For operational use, these should be combined with dynamical climate model outputs (e.g., CMIP6 projections).
            """)
            
        except Exception as e:
            st.error(f"Forecast data not found. Run `python 04_future_forecast.py` first. ({e})")
            st.code("python 04_future_forecast.py", language="bash")

    # ---------------------------------------------------------
    # Tab 9: Forecast Percentage
    # ---------------------------------------------------------
    elif tab_selection == "Forecast Percentage":
        st.title("📊 Forecast Percentage Deviation")
        st.markdown("""
        This tab calculates the percentage deviation of the forecasted Future SPI (`X`) relative to two different baselines:
        
        1. **Average Forecasted SPI (-0.142)**: Compares each year against the overall expected average for 2024-2030.
        2. **Mild Drought Threshold (-0.100)**: Compares each year against a fixed mild drought baseline.
        
        **Formula: Percentage = ((X - Baseline) / |Baseline|) × 100**
        """)
        
        try:
            # Load forecast data to find X and X_bar
            forecast_df = pd.read_csv("outputs/results/forecast_2024_2030.csv")
            
            # Load historical data to calibrate SPI -> Rainfall Deviation
            df_hist = pd.read_csv("data/processed_features.csv", parse_dates=['time'])
            df_hist['year'] = df_hist['time'].dt.year
            yearly_hist = df_hist.groupby('year').agg(
                SPI_mean=('SPI', 'mean'),
                Rainfall_total=('Rainfall', 'sum')
            ).reset_index()
            
            mean_yearly_rf = yearly_hist['Rainfall_total'].mean()
            yearly_hist['RF_pct_dev'] = ((yearly_hist['Rainfall_total'] - mean_yearly_rf) / mean_yearly_rf) * 100
            
            # Calibrate SPI to Rainfall % Deviation (multiplier alpha)
            # Formula: RF_pct_dev = alpha * SPI_mean
            alpha = np.sum(yearly_hist['SPI_mean'] * yearly_hist['RF_pct_dev']) / np.sum(yearly_hist['SPI_mean'] ** 2)
            
            # Baseline 1: Average Forecast (SPI)
            x_bar_avg = forecast_df['Ensemble_SPI'].mean()
            # Baseline 2: Mild Drought Threshold (SPI)
            x_bar_mild = -0.100
            
            col1, col2, col3 = st.columns(3)
            col1.info(f"**Baseline 1 (Avg Forecast SPI):** {x_bar_avg:.3f}")
            col2.warning(f"**Baseline 2 (Mild Threshold SPI):** {x_bar_mild:.3f}")
            col3.success(f"**Calibrated LPA Multiplier:** {alpha:.2f}")
            
            # Select Calculation Method
            calc_method = st.radio(
                "Select Deviation/Percentage Calculation Method:",
                [
                    "IMD Monsoon Rainfall Deviation from LPA (%): SPI * Climatological Multiplier (Recommended)",
                    "Standard Deviation Shift (%) [Z-scores]: (SPI - Baseline) * 100",
                    "Traditional Percentage Deviation (SPI units): ((SPI - Baseline) / |Baseline|) * 100",
                    "Absolute Difference (SPI units): SPI - Baseline"
                ]
            )
            
            # Calculate values based on selected method
            if calc_method.startswith("IMD Monsoon Rainfall Deviation"):
                # Predicted deviation from Forecast Avg and Mild Threshold
                forecast_df['vs Forecast Avg'] = (forecast_df['Ensemble_SPI'] - x_bar_avg) * alpha
                forecast_df['vs Mild Threshold'] = (forecast_df['Ensemble_SPI'] - x_bar_mild) * alpha
                format_dict = {
                    'Ensemble_SPI': '{:.3f}',
                    'vs Forecast Avg': '{:+.2f}%',
                    'vs Mild Threshold': '{:+.2f}%'
                }
                y_label = "Rainfall Deviation from LPA (%)"
                title_suffix = "Rainfall Deviation from LPA"
                is_percentage = True
            elif calc_method.startswith("Standard Deviation Shift"):
                forecast_df['vs Forecast Avg'] = (forecast_df['Ensemble_SPI'] - x_bar_avg) * 100
                forecast_df['vs Mild Threshold'] = (forecast_df['Ensemble_SPI'] - x_bar_mild) * 100
                format_dict = {
                    'Ensemble_SPI': '{:.3f}',
                    'vs Forecast Avg': '{:+.2f}%',
                    'vs Mild Threshold': '{:+.2f}%'
                }
                y_label = "Standard Deviation Shift (%)"
                title_suffix = "Standard Deviation Shift"
                is_percentage = True
            elif calc_method.startswith("Traditional Percentage"):
                forecast_df['vs Forecast Avg'] = ((forecast_df['Ensemble_SPI'] - x_bar_avg) / abs(x_bar_avg)) * 100
                forecast_df['vs Mild Threshold'] = ((forecast_df['Ensemble_SPI'] - x_bar_mild) / abs(x_bar_mild)) * 100
                format_dict = {
                    'Ensemble_SPI': '{:.3f}',
                    'vs Forecast Avg': '{:+.2f}%',
                    'vs Mild Threshold': '{:+.2f}%'
                }
                y_label = "Percentage Deviation (%)"
                title_suffix = "Percentage Deviation (SPI units)"
                is_percentage = True
            else:
                forecast_df['vs Forecast Avg'] = forecast_df['Ensemble_SPI'] - x_bar_avg
                forecast_df['vs Mild Threshold'] = forecast_df['Ensemble_SPI'] - x_bar_mild
                format_dict = {
                    'Ensemble_SPI': '{:.3f}',
                    'vs Forecast Avg': '{:+.3f}',
                    'vs Mild Threshold': '{:+.3f}'
                }
                y_label = "Absolute Difference (SPI)"
                title_suffix = "Absolute Difference"
                is_percentage = False
            
            display_df = forecast_df[['Year', 'Ensemble_SPI', 'vs Forecast Avg', 'vs Mild Threshold']].copy()
            display_df['Year'] = display_df['Year'].astype(int)
            
            st.subheader("Year-by-Year Deviation")
            st.dataframe(display_df.style.format(format_dict), width="stretch")
            
            st.subheader("Deviation Plot")
            fig, ax = plt.subplots(figsize=(12, 5))
            
            # Plot Baseline 1 (Purple)
            ax.plot(display_df['Year'], display_df['vs Forecast Avg'], marker='o', color='purple', linewidth=2, label=f'vs Avg Forecast ({x_bar_avg:.3f})')
            # Plot Baseline 2 (Orange)
            ax.plot(display_df['Year'], display_df['vs Mild Threshold'], marker='s', color='darkorange', linewidth=2, label=f'vs Mild Threshold ({x_bar_mild:.3f})')
            
            ax.axhline(0, color='black', linewidth=1, linestyle='--')
            
            # Alternate annotation positions to avoid clutter
            offsets = [(0, 12), (0, -18), (0, 12), (0, -18), (0, 12), (0, -18), (0, 12)]
            for i, val in enumerate(display_df['vs Mild Threshold']):
                text = f"{val:+.1f}%" if is_percentage else f"{val:+.3f}"
                ax.annotate(text, 
                            (display_df['Year'].iloc[i], display_df['vs Mild Threshold'].iloc[i]),
                            textcoords="offset points", xytext=offsets[i % len(offsets)], 
                            ha='center', fontsize=10, fontweight='bold', color='darkorange',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', edgecolor='darkorange', alpha=0.8))
                            
            ax.set_xlabel('Year', fontweight='bold')
            ax.set_ylabel(y_label, fontweight='bold')
            ax.set_title(f'Future Forecast Deviation ({title_suffix})', fontweight='bold')
            ax.set_xticks(display_df['Year'])
            ax.grid(linestyle=':', alpha=0.5)
            ax.legend()
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error calculating percentages: {e}")
            st.info("Make sure you have run the data preprocessing and forecast scripts.")

