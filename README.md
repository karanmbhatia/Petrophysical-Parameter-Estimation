# Petrophysical-Parameter-Estimation
A well log analysis project using machine learning to predict porosity (phi), water saturation (Sw), and identify hydrocarbon payzones. Includes model training notebooks and an interactive Streamlit web application for visualization and prediction.

##  Project Overview
- **LAS File Integration**: Extract and integrate `.las` well log files into a structured dataset.  
- **Porosity & Saturation Prediction**: Train ML models to predict porosity and water saturation from well logs.  
- **Payzone Classification**: Train a classification model to detect payzones (hydrocarbon-bearing zones).  
- **Web Application**: Deploy models using **Streamlit** with interactive data visualization and prediction capabilities.  

##  Repository Structure
├── notebooks/
│ ├── las_files.ipynb # Extract & preprocess LAS well log files
│ ├── final.ipynb # Train models for porosity (phi) and saturation (sw)
│ ├── payzone_classifier.ipynb # Train payzone classification model
│
├── app/
│ ├── Porosity_and_Saturation.py # Streamlit app for phi & sw prediction [Web UI]
│ ├── 1_Payzone_Prediction.py # Streamlit app for payzone detection [Web UI]
│
├── models/ # Saved models (scaler.pkl, ensemble_phi.pkl, ensemble_sw.pkl, payzone_model.pkl)
│
├── requirements.txt # Project dependencies
├── README.md # Project documentation


## ⚙️ Workflow
### 1. **Data Preparation**
- Used **lasio** to extract logs from `.las` files.  
- Preprocessed and integrated well logs into structured CSV files.  

### 2. **Model Training**
- **Porosity (φ) & Water Saturation (Sw):**
  - Trained ensemble regression models (`Random Forest`, `XGBoost`, `KNN`).  
  - Outputs stored as `ensemble_phi.pkl` and `ensemble_sw.pkl`.  
- **Payzone Classification:**
  - Trained a binary classifier to detect hydrocarbon-bearing zones.  
  - Model saved as `payzone_model.pkl`.  

### 3. **Web Application**
- Built using **Streamlit**.  
- Two main modules:  
  - `Porosity_and_Saturation.py`: Upload CSV logs → Predict φ & Sw → Visualize with depth plots.  
  - `1_Payzone_Prediction.py`: Upload logs → Predict payzones → Highlight intervals on depth plots.  
- Interactive plots, error metrics (MAE, RMSE, Correlation), and CSV downloads.  

## 🚀 How to Run
### 1. Clone Repository
```bash
git clone https://github.com/your-username/well-log-analysis.git
cd well-log-analysis
