import streamlit as st
import pickle
import pandas as pd
import numpy as np
import warnings
import sklearn
from datetime import datetime

# Suppress the UserWarning from scikit-learn when loading the model
warnings.filterwarnings('ignore', category=UserWarning)

# --- App Title and Introduction ---
st.set_page_config(page_title="BigMart Sales Predictor", layout="centered")

st.title("🛒 BigMart Sales Prediction App")
st.markdown("Enter the details of the item and outlet to predict its sales.")

# --- Load the Model ---
@st.cache_resource
def load_model_and_check_version():
    """
    Loads the trained model from the pickle file and checks the scikit-learn version.
    """
    try:
        with open("bigmart_best_model.pkl", "rb") as f:
            model_pipeline, model_sklearn_version = pickle.load(f)
        
        # Check for potential version mismatch
        current_sklearn_version = sklearn.__version__
        if model_sklearn_version != current_sklearn_version:
            st.warning(
                f"⚠️ Model was trained with scikit-learn version {model_sklearn_version}. "
                f"Your current version is {current_sklearn_version}. "
                "This might lead to prediction errors. "
                "For best results, use a compatible version."
            )
        return model_pipeline
    except FileNotFoundError:
        st.error("Model file `bigmart_best_model.pkl` not found. Please ensure it is in the same directory.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

# Load the model and get a handle to it
model = load_model_and_check_version()
if model is None:
    st.stop()

# --- Input Features ---
st.sidebar.header("Item and Outlet Details")

# Input for Item features
item_identifier = st.sidebar.selectbox("Item Identifier", ["FDW58", "FDP10", "FDN15", "FDY07", "NCD19"])
item_weight = st.sidebar.number_input("Item Weight (kg)", min_value=1.0, max_value=25.0, value=12.19, step=0.01)
item_fat_content = st.sidebar.selectbox("Item Fat Content", ["Low Fat", "Regular", "LF", "reg", "low fat"])
item_visibility = st.sidebar.number_input("Item Visibility (%)", min_value=0.0, max_value=0.3, value=0.07, step=0.001)
item_type = st.sidebar.selectbox("Item Type", [
    "Soft Drinks", "Dairy", "Meat", "Fruits and Vegetables", "Household", 
    "Baking Goods", "Snack Foods", "Frozen Foods", "Canned", "Breads", 
    "Health and Hygiene", "Starchy Foods", "Hard Drinks", "Others", 
    "Seafood", "Breakfast"
])
item_mrp = st.sidebar.number_input("Item MRP (₹)", min_value=50.0, max_value=300.0, value=150.93, step=0.01)

# Input for Outlet features
outlet_identifier = st.sidebar.selectbox("Outlet Identifier", ["OUT027", "OUT013", "OUT049", "OUT010", "OUT046"])
outlet_establishment_year = st.sidebar.number_input("Outlet Establishment Year", min_value=1980, max_value=2015, value=2000, step=1)
outlet_size = st.sidebar.selectbox("Outlet Size", ["Small", "Medium", "High"])
outlet_location_type = st.sidebar.selectbox("Outlet Location Type", ["Tier 1", "Tier 2", "Tier 3"])
outlet_type = st.sidebar.selectbox("Outlet Type", ["Supermarket Type1", "Grocery Store", "Supermarket Type2", "Supermarket Type3"])

# --- Prediction Button ---
if st.button("Predict Sales", help="Click to get the sales prediction"):
    # Create a DataFrame from the input features
    input_data = pd.DataFrame([{
        "Item_Identifier": item_identifier,
        "Item_Weight": item_weight,
        "Item_Fat_Content": item_fat_content,
        "Item_Visibility": item_visibility,
        "Item_Type": item_type,
        "Item_MRP": item_mrp,
        "Outlet_Identifier": outlet_identifier,
        "Outlet_Establishment_Year": outlet_establishment_year,
        "Outlet_Size": outlet_size,
        "Outlet_Location_Type": outlet_location_type,
        "Outlet_Type": outlet_type
    }])

    # Add the 'Outlet_Age' feature, which is required by the model
    current_year = datetime.now().year
    input_data['Outlet_Age'] = current_year - input_data['Outlet_Establishment_Year']
    
    # Drop the original 'Outlet_Establishment_Year' if the pipeline doesn't expect it
    input_data = input_data.drop('Outlet_Establishment_Year', axis=1, errors='ignore')

    try:
        # Make a prediction
        prediction = model.predict(input_data)[0]

        # Display the result
        st.subheader("Predicted Sales")
        st.metric(label="Predicted Item Outlet Sales", value=f"₹ {prediction:.2f}")

        # Add a success message with an animation
        st.balloons()
        st.success("Prediction generated successfully!")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")


