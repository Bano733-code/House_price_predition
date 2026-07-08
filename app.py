import streamlit as st
import numpy as np
import joblib


# Page configuration
st.set_page_config(
    page_title="Housing Price Prediction",
    page_icon="🏠",
    layout="centered"
)


# Title
st.title("🏠 California Housing Price Prediction")

st.write(
    "Enter house details below to predict the estimated house price."
)


# Load model and scaler

@st.cache_resource
def load_model():

    model = joblib.load(
        "housing_price_model.pkl"
    )

    scaler = joblib.load(
        "scaler.pkl"
    )

    return model, scaler


model, scaler = load_model()



# User Inputs

longitude = st.number_input(
    "Longitude",
    value=-122.23
)

latitude = st.number_input(
    "Latitude",
    value=37.88
)


housing_age = st.number_input(
    "Housing Median Age",
    value=30
)


total_rooms = st.number_input(
    "Total Rooms",
    value=5000
)


total_bedrooms = st.number_input(
    "Total Bedrooms",
    value=1000
)


population = st.number_input(
    "Population",
    value=1500
)


households = st.number_input(
    "Households",
    value=500
)


median_income = st.number_input(
    "Median Income",
    value=5.0
)



# Prediction

if st.button("Predict House Price"):

    input_data = np.array(
        [
            [
                longitude,
                latitude,
                housing_age,
                total_rooms,
                total_bedrooms,
                population,
                households,
                median_income
            ]
        ]
    )


    # Apply scaling

    input_scaled = scaler.transform(
        input_data
    )


    prediction = model.predict(
        input_scaled
    )


    price = prediction[0]


    st.success(
        f"Estimated House Price: ${price:,.2f}"
    )


    st.info(
        "Prediction generated using Random Forest Regression model."
    )
