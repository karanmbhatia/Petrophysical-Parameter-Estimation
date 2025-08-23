import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib


st.set_page_config(
    page_title="PHI & SW Predictor",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("<h1 style='text-align: center; color: white;'>Porosity & Saturation Prediction</h1>", unsafe_allow_html=True)
st.markdown("<style>body { background-color: #0e1117; color: white; }</style>", unsafe_allow_html=True)


input_cols = ['DEPT', 'RHOC', 'GR', 'RILM', 'RLL3', 'RILD', 'MN', 'CNLS']

# Model Loading
@st.cache_resource
def load_models():
    try:
        scaler = joblib.load(r"C:\Users\Admin\OneDrive - Manipal University Jaipur\Desktop\your_app\models\scaler.pkl")
        phi_model = joblib.load(r"C:\Users\Admin\OneDrive - Manipal University Jaipur\Desktop\your_app\models\ensemble_phi.pkl")
        sw_model = joblib.load(r"C:\Users\Admin\OneDrive - Manipal University Jaipur\Desktop\your_app\models\ensemble_sw.pkl")
        return scaler, phi_model, sw_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

scaler, phi_model, sw_model = load_models()


if 'uploaded_files_data' not in st.session_state:
    st.session_state.uploaded_files_data = {}
if 'selected_well' not in st.session_state:
    st.session_state.selected_well = None

# Sidebar Layout

# 1. File Upload Section
st.sidebar.markdown("### 📁 Upload Well Log CSV Files")
uploaded_files = st.sidebar.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)

# Process uploaded files
if uploaded_files and scaler is not None:
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state.uploaded_files_data:
            df = pd.read_csv(uploaded_file)
            
            
            missing_cols = [col for col in input_cols if col not in df.columns]
            if not missing_cols:
                
                X = df[input_cols]
                X_scaled = scaler.transform(X)
                df['phi_pred'] = phi_model.predict(X_scaled)
                df['sw_pred'] = sw_model.predict(X_scaled)
                
                
                df = df.sort_values(by='DEPT')
                
                
                st.session_state.uploaded_files_data[uploaded_file.name] = df
            else:
                st.error(f"File {uploaded_file.name} missing required columns: {missing_cols}")

# 2. Well Selection Section
if st.session_state.uploaded_files_data:
    st.sidebar.markdown("###  Select Well to Analyze")
    well_names = list(st.session_state.uploaded_files_data.keys())
    selected_well = st.sidebar.selectbox("Choose a well:", well_names, key="well_selector")
    st.session_state.selected_well = selected_well



#Main Content

if st.session_state.selected_well and st.session_state.selected_well in st.session_state.uploaded_files_data:
    df = st.session_state.uploaded_files_data[st.session_state.selected_well]
    
    st.markdown(f"##  Analysis for: **{st.session_state.selected_well}**")
    
    
    phi_actual_col = None
    sw_actual_col = None
    
    phi_possible_names = ['PHI', 'PHIT', 'PHIE', 'POROSITY', 'phi', 'phit', 'phie', 'porosity']
    sw_possible_names = ['SW', 'SW_ACTUAL', 'SWAT', 'sw', 'sw_actual', 'swat', 'Sw', 'SW_TRUE']
    
    for col in df.columns:
        if col.upper() in [name.upper() for name in phi_possible_names]:
            phi_actual_col = col
            break
            
    for col in df.columns:
        if col.upper() in [name.upper() for name in sw_possible_names]:
            sw_actual_col = col
            break

    # seperate well plots
    plot_col1, plot_col2 = st.columns(2)
    
    with plot_col1:
        with st.expander(" **PHI Log Plot** (Click to expand)", expanded=False):
            fig_phi, ax_phi = plt.subplots(1, 1, figsize=(8, 12))
            fig_phi.patch.set_facecolor('#0e1117')
            
            if phi_actual_col:
                ax_phi.plot(df[phi_actual_col], df['DEPT'], color='red', linewidth=2, label=f'Actual PHI', alpha=0.9)
            ax_phi.plot(df['phi_pred'], df['DEPT'], color='yellow', linewidth=2, label='Predicted PHI', alpha=0.9)
            
            ax_phi.invert_yaxis()
            ax_phi.set_xlabel("PHI", color='white', fontsize=14, fontweight='bold')
            ax_phi.set_ylabel("Depth (ft)", color='white', fontsize=14, fontweight='bold')
            ax_phi.set_title("PHI Log", color='yellow', fontsize=18, fontweight='bold', pad=20)
            ax_phi.tick_params(axis='both', colors='white', labelsize=12)
            ax_phi.grid(True, alpha=0.4, color='white', linewidth=0.5)
            #ax_phi.legend(loc='upper right', facecolor='#1e1e1e', edgecolor='white', fontsize=12, framealpha=0.9)

            legend = ax_phi.legend(loc='upper right', facecolor='#1e1e1e', edgecolor='white', fontsize=12, framealpha=0.9)
            for text in legend.get_texts():
                text.set_color('white')
            
            # Set better x-axis limits for PHI
            phi_min = df['phi_pred'].min()
            phi_max = df['phi_pred'].max()
            if phi_actual_col:
                phi_min = min(phi_min, df[phi_actual_col].min())
                phi_max = max(phi_max, df[phi_actual_col].max())
            
            phi_range = phi_max - phi_min
            ax_phi.set_xlim(phi_min - 0.15*phi_range, phi_max + 0.15*phi_range)
            
            for spine in ax_phi.spines.values():
                spine.set_color('white')
                spine.set_linewidth(2)
            ax_phi.set_facecolor('#0e1117')
            
            plt.tight_layout()
            st.pyplot(fig_phi)
    
    with plot_col2:
        with st.expander(" **SW Log Plot** (Click to expand)", expanded=False):
            fig_sw, ax_sw = plt.subplots(1, 1, figsize=(8, 12))
            fig_sw.patch.set_facecolor('#0e1117')
            
            if sw_actual_col:
                ax_sw.plot(df[sw_actual_col], df['DEPT'], color='red', linewidth=2, label=f'Actual SW', alpha=0.9)
            ax_sw.plot(df['sw_pred'], df['DEPT'], color='cyan', linewidth=2, label='Predicted SW', alpha=0.9)
            
            ax_sw.invert_yaxis()
            ax_sw.set_xlabel("SW", color='white', fontsize=14, fontweight='bold')
            ax_sw.set_ylabel("Depth (ft)", color='white', fontsize=14, fontweight='bold')
            ax_sw.set_title("SW Log", color='cyan', fontsize=18, fontweight='bold', pad=20)
            ax_sw.tick_params(axis='both', colors='white', labelsize=12)
            ax_sw.grid(True, alpha=0.4, color='white', linewidth=0.5)
            #ax_sw.legend(loc='upper right', facecolor='#1e1e1e', edgecolor='white', fontsize=12, framealpha=0.9)
            
            legend = ax_sw.legend(loc='upper right', facecolor='#1e1e1e', edgecolor='white', fontsize=12, framealpha=0.9)
            for text in legend.get_texts():
                text.set_color('white')
            
            sw_min = df['sw_pred'].min()
            sw_max = df['sw_pred'].max()
            if sw_actual_col:
                sw_min = min(sw_min, df[sw_actual_col].min())
                sw_max = max(sw_max, df[sw_actual_col].max())
            
            sw_range = sw_max - sw_min
            ax_sw.set_xlim(sw_min - 0.15*sw_range, sw_max + 0.15*sw_range)
            
            for spine in ax_sw.spines.values():
                spine.set_color('white')
                spine.set_linewidth(2)
            ax_sw.set_facecolor('#0e1117')
            
            plt.tight_layout()
            st.pyplot(fig_sw)

    
    st.markdown("###  Data Information")
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        if phi_actual_col:
            st.success(f"✅ Actual PHI column found: **{phi_actual_col}**")
        else:
            st.warning("⚠️ No actual PHI column found in uploaded data")
            st.info("Expected column names: " + ", ".join(phi_possible_names))
    
    with info_col2:
        if sw_actual_col:
            st.success(f"✅ Actual SW column found: **{sw_actual_col}**")
        else:
            st.warning("⚠️ No actual SW column found in uploaded data")
            st.info("Expected column names: " + ", ".join(sw_possible_names))

    # Display statistics
    if phi_actual_col or sw_actual_col:
        st.markdown("###  Prediction Statistics")
        
        if phi_actual_col:
            phi_mae = abs(df['phi_pred'] - df[phi_actual_col]).mean()
            phi_rmse = ((df['phi_pred'] - df[phi_actual_col]) ** 2).mean() ** 0.5
            phi_corr = df['phi_pred'].corr(df[phi_actual_col])
            
            st.markdown("**PHI Statistics:**")
            col1, col2, col3 = st.columns(3)
            col1.metric("MAE", f"{phi_mae:.4f}")
            col2.metric("RMSE", f"{phi_rmse:.4f}")
            col3.metric("Correlation", f"{phi_corr:.4f}")
        
        if sw_actual_col:
            sw_mae = abs(df['sw_pred'] - df[sw_actual_col]).mean()
            sw_rmse = ((df['sw_pred'] - df[sw_actual_col]) ** 2).mean() ** 0.5
            sw_corr = df['sw_pred'].corr(df[sw_actual_col])
            
            st.markdown("**SW Statistics:**")
            col1, col2, col3 = st.columns(3)
            col1.metric("MAE", f"{sw_mae:.4f}")
            col2.metric("RMSE", f"{sw_rmse:.4f}")
            col3.metric("Correlation", f"{sw_corr:.4f}")

    # Data preview 
    with st.expander("View Uploaded Data Preview"):
        st.dataframe(df, use_container_width=True)

    # Download predictions
    st.markdown("### 💾 Download Results")
    csv_download = df.to_csv(index=False)
    st.download_button(
        label=" Download Predictions as CSV",
        data=csv_download,
        file_name=f"{st.session_state.selected_well.replace('.csv', '')}_predictions.csv",
        mime="text/csv"
    )

# Manual Prediction 
st.markdown("---")
st.markdown("###  Manual Prediction Input")

if scaler is not None:
    
    if 'clear_counter' not in st.session_state:
        st.session_state.clear_counter = 0
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None

    
    col1, col2 = st.columns(2)
    
    
    current_inputs = {}
    
    with col1:
        for col in input_cols[:4]:
            
            value = st.number_input(
                f"{col}", 
                value=None,
                placeholder="Enter value...",
                key=f"input_{col}_{st.session_state.clear_counter}", 
                format="%.6f"
            )
            current_inputs[col] = value

    with col2:
        for col in input_cols[4:]:
            
            value = st.number_input(
                f"{col}", 
                value=None,
                placeholder="Enter value...",
                key=f"input_{col}_{st.session_state.clear_counter}", 
                format="%.6f"
            )
            current_inputs[col] = value

    # Predict & Clear buttons
    col_a, col_b = st.columns([1, 1])

    with col_a:
        if st.button(" Predict ", type="primary"):
            # Check if all fields are filled
            all_filled = True
            for col in input_cols:
                if current_inputs[col] is None:
                    all_filled = False
                    break
            
            if all_filled:
                # Make prediction only when button is clicked 
                input_df = pd.DataFrame([current_inputs])
                input_scaled = scaler.transform(input_df)
                phi_val = phi_model.predict(input_scaled)[0]
                sw_val = sw_model.predict(input_scaled)[0]
                
                st.session_state.prediction_result = {
                    'phi': phi_val,
                    'sw': sw_val
                }
            else:
                st.error("⚠️ Please fill in all input fields before predicting.")

    with col_b:
        if st.button("Clear Inputs"):
            
            st.session_state.clear_counter += 1
            st.session_state.prediction_result = None
            st.rerun()

    # Show result only if prediction was made
    if st.session_state.prediction_result:
        st.markdown("###  Prediction Result")
        res1, res2 = st.columns(2)
        res1.metric("Predicted PHI", f"{st.session_state.prediction_result['phi']:.4f}")
        res2.metric("Predicted SW", f"{st.session_state.prediction_result['sw']:.4f}")

else:
    st.error(" Models could not be loaded. Please check your model files.")