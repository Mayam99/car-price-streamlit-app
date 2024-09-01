# app.py
# Streamlit Car Price Prediction App

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Function to load data
@st.cache_data
def load_data():
    # Load the dataset from the local directory or GitHub if deployed
    # Provide the correct file path if running locally
    # If running on Streamlit Cloud, the file needs to be accessible via URL
    data_url = "CarPricesPrediction.csv"  # Adjust this if required
    data = pd.read_csv(data_url)
    return data

# Function to load the saved model
@st.cache_resource
def load_model():
    # Provide the correct file path for your saved model
    model_url = "car_price_model.pkl"  # Adjust this if required
    with open(model_url, 'rb') as file:
        model = pickle.load(file)
    return model

# Function to get user input
def user_input_features():
    st.sidebar.header('Input Car Details')
    make = st.sidebar.selectbox('Make', ('Ford', 'Toyota', 'Chevrolet'))
    model = st.sidebar.selectbox('Model', ('Silverado', 'Civic'))
    year = st.sidebar.slider('Year', 2000, 2023, 2020)
    mileage = st.sidebar.slider('Mileage', 0, 100000, 15000)
    condition = st.sidebar.selectbox('Condition', ('Excellent', 'Good', 'Fair'))

    # Create a DataFrame based on user input
    data = {
        'Make': make,
        'Model': model,
        'Year': year,
        'Mileage': mileage,
        'Condition': condition
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Main function for the app
def main():
    st.title('Car Price Prediction App')
    st.write("""
    This app predicts the price of a car based on user input. Adjust the car details in the sidebar to get a price prediction.
    """)

    # Load data and model
    df = load_data()
    model = load_model()

    # Display the first few rows of the dataset
    st.subheader('Dataset Sample')
    st.write(df.head())

    # Get user input
    input_df = user_input_features()

    # Display the user input features
    st.subheader('User Input Features')
    st.write(input_df)

    # Data preprocessing: encoding and scaling as done during training
    scaler = StandardScaler()
    df[['Year', 'Mileage']] = scaler.fit_transform(df[['Year', 'Mileage']])
    input_df[['Year', 'Mileage']] = scaler.transform(input_df[['Year', 'Mileage']])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

    # Predict the price
    prediction = model.predict(input_df)

    # Display the predicted price
    st.subheader('Predicted Car Price')
    st.write(f"Estimated Price: ${prediction[0]:,.2f}")

if __name__ == '__main__':
    main()
