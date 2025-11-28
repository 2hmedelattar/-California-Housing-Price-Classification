import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------------------
# 1. Page Config (MUST BE THE FIRST STREAMLIT COMMAND)
# -------------------------------------------
st.set_page_config(page_title="California Housing Prediction", page_icon="🏡")

# -------------------------------------------
# 2. Load Model and Scaler
# -------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load('housing_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_artifacts()
except FileNotFoundError:
    st.error("⚠️ الملفات 'housing_model.pkl' و 'scaler.pkl' مش موجودين! لازم تشغل كود الحفظ في النوت بوك الأول وتتأكد إنهم في نفس الفولدر.")
    st.stop()
except Exception as e:
    st.error(f"⚠️ حصلت مشكلة أثناء تحميل الموديل: {e}")
    st.stop()

# -------------------------------------------
# 3. UI Structure (Title & Sidebar)
# -------------------------------------------
st.title("🏡 California Housing Price Classifier")
st.markdown("Enter the house details below to predict if the price category is **Low**, **Medium**, or **High**.")

st.sidebar.header("User Input Features")

# -------------------------------------------
# 4. User Inputs (The 8 Original Features)
# -------------------------------------------
def user_input_features():
    MedInc = st.sidebar.slider('Median Income (Tens of thousands)', 0.5, 15.0, 3.5)
    HouseAge = st.sidebar.slider('House Age (Years)', 1, 52, 20)
    AveRooms = st.sidebar.slider('Average Rooms', 1.0, 10.0, 5.0)
    AveBedrms = st.sidebar.slider('Average Bedrooms', 0.5, 5.0, 1.0)
    Population = st.sidebar.slider('Population', 100, 10000, 1000)
    AveOccup = st.sidebar.slider('Average Occupancy', 1.0, 6.0, 3.0)
    Latitude = st.sidebar.number_input('Latitude', 32.0, 42.0, 34.05, format="%.2f")
    Longitude = st.sidebar.number_input('Longitude', -125.0, -114.0, -118.24, format="%.2f")
    
    data = {
        'MedInc': MedInc,
        'HouseAge': HouseAge,
        'AveRooms': AveRooms,
        'AveBedrms': AveBedrms,
        'Population': Population,
        'AveOccup': AveOccup,
        'Latitude': Latitude,
        'Longitude': Longitude
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Display input data
st.subheader("Current Input Parameters:")
st.write(input_df)

# -------------------------------------------
# 5. Feature Engineering (Replicating Notebook Logic)
# -------------------------------------------
def preprocess_data(df):
    df = df.copy()
    
    # 1. Feature Engineering Calculations
    df['Bedroom_Ratio'] = df['AveBedrms'] / df['AveRooms']
    df['Rooms_Per_Person'] = df['AveRooms'] / df['AveOccup']
    
    # Rotated Coordinates
    df['Rotated_Coords_1'] = df['Latitude'] + df['Longitude']
    df['Rotated_Coords_2'] = df['Latitude'] - df['Longitude']
    
    # Extra ratios
    df['Bedrooms_per_Room'] = df['AveBedrms'] / df['AveRooms']
    df['Population_per_Household'] = df['Population'] / df['AveOccup']
    df['Rooms_per_Household'] = df['AveRooms']
    
    # Distance Functions
    def calculate_distance(lat, lon, city_lat, city_lon):
        return np.sqrt((lat - city_lat) ** 2 + (lon - city_lon) ** 2)

    df['Dist_to_LA'] = calculate_distance(df['Latitude'], df['Longitude'], 34.05, -118.24)
    df['Dist_to_SF'] = calculate_distance(df['Latitude'], df['Longitude'], 37.77, -122.41)
    
    # 2. Reorder columns
    expected_cols = [
        'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude',
        'Bedroom_Ratio', 'Rooms_Per_Person', 'Rotated_Coords_1', 'Rotated_Coords_2',
        'Bedrooms_per_Room', 'Population_per_Household', 'Rooms_per_Household',
        'Dist_to_LA', 'Dist_to_SF'
    ]
    
    # Ensure correct column order
    df = df[expected_cols]
    
    return df

# Process the input
processed_df = preprocess_data(input_df)

# -------------------------------------------
# 6. Prediction
# -------------------------------------------
if st.button('Predict Price Category', type='primary'): # type='primary' makes the button stand out
    # Scale the data using the loaded scaler
    scaled_data = scaler.transform(processed_df)
    
    # Make Prediction
    prediction = model.predict(scaled_data)[0]
    
    # Mapping output
    label_map = {0: 'Low Price 📉', 1: 'Medium Price 📊', 2: 'High Price 💰'}
    result = label_map.get(prediction, "Unknown")
    
    # Display Result
    st.markdown("---")
    st.subheader("Prediction Result:")
    
    if prediction == 2:
        st.success(f"### The house is classified as: {result}")
    elif prediction == 1:
        st.info(f"### The house is classified as: {result}")
    else:
        st.warning(f"### The house is classified as: {result}")
