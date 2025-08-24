# Petrophysical-Parameter-Estimation
A well log analysis project using machine learning to predict porosity (phi), water saturation (Sw), and identify hydrocarbon payzones. Includes model training notebooks and an interactive Streamlit web application for visualization and prediction.

## Project Overview  
- **Data Preprocessing**: Clean, preprocess, and prepare raw well log data for model training.  
- **LAS File Integration**: Extract and integrate `.las` well log files into a structured dataset.  
- **Porosity & Saturation Prediction**: Train ML models to predict porosity and water saturation from well logs.  
- **Payzone Classification**: Train a classification model to detect payzones (hydrocarbon-bearing zones).  
- **Web Application**: Deploy models using Streamlit with interactive data visualization and prediction capabilities.  

## Website UI Demo  
[![Watch the demo](https://img.youtube.com/vi/RMq8df75HsU/0.jpg)](https://youtu.be/RMq8df75HsU)  

##  Workflow
### 1. **Data Preparation**
- Used **lasio** to extract logs from `.las` files.  
- Preprocessed and integrated well logs into structured CSV files.  

### 2. **Model Training**
- **Porosity (φ) and Water Saturation (Sw):**
  - Trained ensemble regression models (`Random Forest`, `XGBoost`, `KNN`).  
  - Outputs stored as `ensemble_phi.pkl` and `ensemble_sw.pkl`.  
- **Payzone Classification:**
  - Trained a binary classifier to detect hydrocarbon-bearing zones.  
  - Model saved as `payzone_model.pkl`.  

### 3. **Web Application**
- Built using **Streamlit**.  
- Two main modules:  
  - `Porosity_and_Saturation.py`: Upload CSV logs → Predict Porosity(φ) and Water Saturation(Sw) → Visualize with depth plots.  
  - `1_Payzone_Prediction.py`: Upload logs → Predict payzones → Highlight intervals on depth plots.  
- Interactive plots, error metrics (MAE, RMSE, Correlation), and CSV downloads.  

