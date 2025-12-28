import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(
    page_title="California House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

st.title("House Price Prediction")
st.write(
    "Fill the details below **(based on the California Housing Prices dataset ranges).**"
)

# Load model & pipeline
@st.cache_resource
def load_model():
    pipeline = joblib.load("pipeline.pkl")
    model = joblib.load("model.pkl")
    return pipeline, model

pipeline, model = load_model()

# -------- Input Form --------
with st.form("house_price_form"):

    longitude = st.number_input(
        "Longitude",
        help="How far west the house is. Values are roughly between -124 and -114 (California, USA)."
    )

    latitude = st.number_input(
        "Latitude",
        help="How far north the house is. Values are roughly between 32 and 42."
    )

    housing_median_age = st.number_input(
        "Housing Median Age (years)",
        min_value=0.0,
        help="Median age of houses in the area (typically between 1 and 52 years)."
    )

    total_rooms = st.number_input(
        "Total Rooms",
        min_value=0.0,
        help="Total number of rooms in all houses in this area/block (e.g., 800 – 4000+)."
    )

    total_bedrooms = st.number_input(
        "Total Bedrooms",
        min_value=0.0,
        help="Total number of bedrooms in all houses in this area/block."
    )

    population = st.number_input(
        "Population",
        min_value=0.0,
        help="Number of people living in this area/block."
    )

    households = st.number_input(
        "Households",
        min_value=0.0,
        help="Number of households (families/flats) in this area/block."
    )

    median_income = st.number_input(
        "Median Income",
        min_value=0.0,
        help="Median income of households in the area (in tens of thousands USD)."
    )

    ocean_proximity = st.selectbox(
        "Ocean Proximity",
        ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"],
        help="Location of the house relative to the ocean."
    )

    submit = st.form_submit_button("Predict House Price")

# -------- Prediction --------
if submit:
    try:
        input_data = pd.DataFrame([{
            "longitude": longitude,
            "latitude": latitude,
            "housing_median_age": housing_median_age,
            "total_rooms": total_rooms,
            "total_bedrooms": total_bedrooms,
            "population": population,
            "households": households,
            "median_income": median_income,
            "ocean_proximity": ocean_proximity
        }])

        X_processed = pipeline.transform(input_data)
        prediction = model.predict(X_processed)[0]

        st.success(f"💰 Estimated House Price: **${prediction:,.2f}**")

    except Exception as e:
        st.error(f"❌ Error: {e}")
