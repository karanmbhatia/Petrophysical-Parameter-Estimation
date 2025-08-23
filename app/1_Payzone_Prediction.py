import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np
import os


st.set_page_config(
    page_title="Payzone Detection",
    page_icon="🛢️",
    layout="wide"
)

st.markdown("<h1 style='text-align: center; color: white;'> Payzone Detection</h1>", unsafe_allow_html=True)
st.markdown("<style>body { background-color: #0e1117; color: white; }</style>", unsafe_allow_html=True)


input_cols = ['DEPT', 'RHOC', 'GR', 'RILM', 'RLL3', 'RILD', 'MN', 'CNLS', 'phi', 'sw']

# Model Loading 
@st.cache_resource
def load_payzone_model():
    """Load only the payzone prediction model"""
    try:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_path, 'models', 'payzone_model.pkl')
        
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            return model
        else:
            return None
    except Exception as e:
        return None


payzone_model = load_payzone_model()


if 'payzone_files_data' not in st.session_state:
    st.session_state.payzone_files_data = {}
if 'selected_payzone_well' not in st.session_state:
    st.session_state.selected_payzone_well = None
if 'show_fullscreen' not in st.session_state:
    st.session_state.show_fullscreen = False

# Sidebar Layout 
# File Upload Section
st.sidebar.markdown("### 📁 Upload Well Log CSV Files")
uploaded_files = st.sidebar.file_uploader(
    "Upload CSV files for payzone analysis", 
    type="csv", 
    accept_multiple_files=True
)

# Process uploaded files
if uploaded_files and payzone_model is not None:
    for uploaded_file in uploaded_files:
        try:
            if uploaded_file.name not in st.session_state.payzone_files_data:
                # Read CSV
                df = pd.read_csv(uploaded_file)
                
                # Remove TARGET column automatically if it exists
                if 'TARGET' in df.columns:
                    df = df.drop(columns=['TARGET'])
                
                # Validate required columns
                missing_cols = [col for col in input_cols if col not in df.columns]
                if missing_cols:
                    st.error(f"❌ File {uploaded_file.name} missing required columns: {missing_cols}")
                    st.info(f"Available columns: {', '.join(df.columns)}")
                    continue
                
                # Check for empty dataframe
                if df.empty:
                    st.error(f"❌ File {uploaded_file.name} is empty")
                    continue
                
                # Remove rows with NaN in required columns
                df_clean = df.dropna(subset=input_cols)
                if len(df_clean) < len(df):
                    rows_removed = len(df) - len(df_clean)
                    df = df_clean
                
                # Sort by depth
                df = df.sort_values(by='DEPT').reset_index(drop=True)
                
                # Make payzone predictions
                X = df[input_cols]
                df['payzone_prediction'] = payzone_model.predict(X)
                
                # Calculate payzone statistics
                payzone_count = (df['payzone_prediction'] == 1).sum()
                st.success(f" Processed {uploaded_file.name} - Found {payzone_count} payzone intervals")
                
                # Store processed data
                st.session_state.payzone_files_data[uploaded_file.name] = df
                
        except Exception as e:
            st.error(f" Error processing {uploaded_file.name}: {str(e)}")

# Well Selection Section
if st.session_state.payzone_files_data:
    st.sidebar.markdown("###  Select Well for Analysis")
    well_names = list(st.session_state.payzone_files_data.keys())
    selected_well = st.sidebar.selectbox("Choose a well:", well_names)
    st.session_state.selected_payzone_well = selected_well


#  Main Content
if st.session_state.selected_payzone_well and st.session_state.selected_payzone_well in st.session_state.payzone_files_data:
    df = st.session_state.payzone_files_data[st.session_state.selected_payzone_well]
    
    st.markdown(f"## Payzone Analysis for: **{st.session_state.selected_payzone_well}**")
    
    # Create columns for plot and fullscreen button
    plot_col, button_col = st.columns([10, 1])
    
    with button_col:
        if st.button("🔍", help="View fullscreen"):
            st.session_state.show_fullscreen = not st.session_state.show_fullscreen
    
    # Determine figure size based on fullscreen state
    if st.session_state.show_fullscreen:
        fig_width, fig_height = 10, 8
    else:
        fig_width, fig_height = 8, 6
    
    # Main Payzone Visualization
    try:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        
        # Get payzone intervals only
        payzone_df = df[df['payzone_prediction'] == 1].copy()
        
        if not payzone_df.empty:
            # Calculate appropriate line thickness
            depth_range = df['DEPT'].max() - df['DEPT'].min()
            if len(df) > 1:
                depth_diffs = np.diff(sorted(df['DEPT'].values))
                line_thickness = np.median(depth_diffs) * 0.8
            else:
                line_thickness = depth_range * 0.01
            
            # Plot yellow horizontal lines for payzones
            for idx, row in payzone_df.iterrows():
                depth = row['DEPT']
                ax.barh(depth, 1, height=line_thickness, left=0, 
                       color='yellow', alpha=0.9, edgecolor='gold', linewidth=1)
            
            # Set depth limits with proper orientation
            depth_min = df['DEPT'].min()
            depth_max = df['DEPT'].max()
            padding = (depth_max - depth_min) * 0.05
            
            # Set y-axis limits - this will show min at top, max at bottom after inversion
            ax.set_ylim(depth_max + padding, depth_min - padding)
            ax.set_xlim(-0.1, 1.1)
            
            # Labels and title
            ax.set_xlabel("Payzone Detection", color='white', fontsize=12, fontweight='bold')
            ax.set_ylabel("Depth (ft)", color='white', fontsize=12, fontweight='bold')
            ax.set_title(f"Payzone Intervals - {len(payzone_df)} zones detected", 
                        color='yellow', fontsize=14, fontweight='bold', pad=15)
            
            # Remove x-axis ticks
            ax.set_xticks([])
            
            # Y-axis formatting
            ax.tick_params(axis='y', colors='white', labelsize=10)
            ax.grid(True, axis='y', alpha=0.3, color='white', linewidth=0.5)
            ax.grid(False, axis='x')
            
        else:
            ax.text(0.5, 0.5, 'No Payzone Intervals Detected', 
                   transform=ax.transAxes, ha='center', va='center',
                   color='white', fontsize=16)
        
        # Style spines
        for spine in ax.spines.values():
            spine.set_color('white')
            spine.set_linewidth(1)
        
        plt.tight_layout()
        
        if st.session_state.show_fullscreen:
            st.pyplot(fig, use_container_width=True)
        else:
            with plot_col:
                st.pyplot(fig)
        
    except Exception as e:
        st.error(f"❌ Error creating plot: {str(e)}")
    
    # Display statistics below the plot
    st.markdown("###  Payzone Statistics")
    
    payzone_count = (df['payzone_prediction'] == 1).sum()
    non_payzone_count = (df['payzone_prediction'] == 0).sum()
    total_count = len(df)
    payzone_percentage = (payzone_count / total_count * 100) if total_count > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Depth Intervals", total_count)
    col2.metric("Payzone Intervals", payzone_count)
    col3.metric("Non-Payzone Intervals", non_payzone_count)
    col4.metric("Payzone Percentage", f"{payzone_percentage:.1f}%")
    
    # Payzone details table
    with st.expander(" **Payzone Intervals Details**", expanded=False):
        payzone_df = df[df['payzone_prediction'] == 1].copy()
        if not payzone_df.empty:
            st.markdown("**Detected Payzone Intervals:**")
            
            # Select relevant columns to display
            display_cols = ['DEPT', 'GR', 'RHOC', 'RILM', 'RLL3', 'RILD', 'phi', 'sw']
            display_cols = [col for col in display_cols if col in payzone_df.columns]
            
            st.dataframe(
                payzone_df[display_cols].round(2).style.highlight_max(axis=0),
                use_container_width=True
            )
            
            # Depth range summary
            st.markdown("**Payzone Depth Summary:**")
            col1, col2, col3 = st.columns(3)
            col1.metric("Shallowest Payzone", f"{payzone_df['DEPT'].min():.0f} ft")
            col2.metric("Deepest Payzone", f"{payzone_df['DEPT'].max():.0f} ft")
            col3.metric("Depth Range", f"{payzone_df['DEPT'].max() - payzone_df['DEPT'].min():.0f} ft")
        else:
            st.info("No payzone intervals detected in this well.")
    
    # Download results
    st.markdown("### 💾 Download Results")
    
    # Prepare download data
    download_df = df.copy()
    download_df['payzone'] = download_df['payzone_prediction']
    
    csv_data = download_df.to_csv(index=False)
    st.download_button(
        label=" Download Complete Results (CSV)",
        data=csv_data,
        file_name=f"{st.session_state.selected_payzone_well.replace('.csv', '')}_payzone_predictions.csv",
        mime="text/csv"
    )
    
    # Download only payzone intervals
    payzone_only_df = df[df['payzone_prediction'] == 1].copy()
    if not payzone_only_df.empty:
        payzone_csv = payzone_only_df.to_csv(index=False)
        st.download_button(
            label=" Download Payzone Intervals Only (CSV)",
            data=payzone_csv,
            file_name=f"{st.session_state.selected_payzone_well.replace('.csv', '')}_payzone_intervals_only.csv",
            mime="text/csv"
        )

elif payzone_model is None:
    st.error(" Payzone model could not be loaded. Please check that 'payzone_model.pkl' exists in the models directory.")
else:
    st.info(" Please upload well log CSV files using the sidebar to begin payzone analysis.")
    
    # Instructions
    st.markdown("""
    ###  How to Use:
    
    1. **Upload CSV Files** 
       - Use the sidebar to upload one or more well log CSV files
       - Files should contain the required columns listed below
       - The TARGET column will be automatically removed
    
    2. **View Results**
       - Select a well from the dropdown to see its payzone analysis
       - Yellow horizontal lines indicate predicted payzone intervals
       - Use the 🔍 button to view the plot in fullscreen
    
    3. **Download Predictions**
       - Download complete results with all predictions
       - Download only the payzone intervals for focused analysis
    
    ### Required Columns:
    Your CSV files must contain these columns:
    - `DEPT` - Depth (ft)
    - `RHOC` - Bulk Density (g/cm³)
    - `GR` - Gamma Ray (API)
    - `RILM` - Medium Resistivity (ohm.m)
    - `RLL3` - Shallow Resistivity (ohm.m)
    - `RILD` - Deep Resistivity (ohm.m)
    - `MN` - Neutron (limestone units)
    - `CNLS` - Neutron (sandstone units)
    - `phi` - Porosity (fraction)
    - `sw` - Water Saturation (fraction)
    
    ###  What are Payzones?
    Payzones are intervals in a well that have the potential to produce hydrocarbons (oil/gas). 
    The model identifies these zones based on the well log measurements.
    
    ###  Visualization:
    - **Yellow lines**: Predicted payzone intervals (potential hydrocarbon zones)
    - **Depth axis**: Increases downward (as in real wells)
    - **Fullscreen**: Click 🔍 to expand the plot for better viewing
    """)