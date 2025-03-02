import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from keras.saving import register_keras_serializable
from datetime import datetime
import matplotlib.pyplot as plt
import os
import logging
import seaborn as sns


# Define Carbon Tax Rate (Taiwan) in NT$ per ton of CO₂
carbon_tax_rate_per_ton = 300  

# Function to Calculate Carbon Tax
def calculate_carbon_tax(emissions, tax_rate_per_ton):
    """
    Calculate the carbon tax for predicted CO₂ emissions.
    :param emissions: CO₂ emissions in kg/100 km
    :param tax_rate_per_ton: Carbon tax rate in NT$ per ton of CO₂
    :return: Carbon tax amount in NT$
    """
    emissions_in_tons = emissions / 1000  # Convert kg to tons
    return emissions_in_tons * tax_rate_per_ton

# Helper Functions for Carbon Tax Calculation
def calculate_annual_tax(emissions_per_km, mileage, tax_rate_per_gram):
    """
    Calculate annual tax based on emissions per km, mileage, and tax rate per gram of CO₂.
    """
    annual_emissions = emissions_per_km * mileage  # Total annual emissions in grams
    annual_tax = annual_emissions * tax_rate_per_gram  # Total tax in TWD
    return annual_tax


def phased_implementation(emissions_per_km, mileage, phase):
    """
    Calculate tax based on the phased implementation logic.
    """
    if phase == 1:
        tax_rate_per_gram = 0.0025  # Phase 1 (Years 1–3)
    elif phase == 2:
        tax_rate_per_gram = 0.005  # Phase 2 (Years 4–6)
    else:
        raise ValueError("Invalid phase. Use 1 for Years 1–3 or 2 for Years 4–6.")
    return calculate_annual_tax(emissions_per_km, mileage, tax_rate_per_gram)


def adjust_tax_rate(emissions_per_km, mileage, vehicle_type):
    """
    Adjust the tax rate based on vehicle type.
    """
    base_tax_rate = 0.005  # Default tax rate in TWD per gram of CO₂
    if vehicle_type == "low-emission":
        tax_rate_per_gram = base_tax_rate * 0.5  # 50% discount
    elif vehicle_type == "high-emission":
        tax_rate_per_gram = base_tax_rate * 1.4  # 40% increase
    else:
        tax_rate_per_gram = base_tax_rate  # Standard rate
    return calculate_annual_tax(emissions_per_km, mileage, tax_rate_per_gram)


# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)

st.write("App started successfully!")

# Register 'mse' as a serializable custom object
@register_keras_serializable(package="Custom", name="mse")
def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Load Models and Preprocessor
@st.cache_resource
def load_models():
    try:
        logging.debug("Loading LSTM model...")
        lstm_model = tf.keras.models.load_model('lstm_model.h5', custom_objects={"mse": mse})
        logging.debug("LSTM model loaded successfully.")

        logging.debug("Loading RNN model...")
        rnn_model = tf.keras.models.load_model('rnn_model.h5', custom_objects={"mse": mse})
        logging.debug("RNN model loaded successfully.")

        logging.debug("Loading Regression model...")
        reg_model = tf.keras.models.load_model('reg_model.h5', custom_objects={"mse": mse})
        logging.debug("Regression model loaded successfully.")

        logging.debug("Loading GRU model...")
        gru_model = tf.keras.models.load_model('gru_model.h5', custom_objects={"mse": mse})
        logging.debug("GRU model loaded successfully.")

        logging.debug("Loading Seq2Seq model...")
        seq2seq_model = tf.keras.models.load_model('seq2seq_model.h5', custom_objects={"mse": mse})
        logging.debug("Seq2Seq model loaded successfully.")

        logging.debug("Loading Transformer model...")
        try:
            transformer_model = tf.keras.models.load_model('transformer_model.h5', custom_objects={"mse": mse})
            logging.debug("Transformer model loaded successfully.")
        except Exception as transformer_error:
            logging.error(f"Error loading Transformer model: {transformer_error}")
            st.error(f"Error loading Transformer model: {transformer_error}")
            transformer_model = None  # Fallback if Transformer fails

        logging.debug("Loading preprocessor...")
        with open('preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        logging.debug("Preprocessor loaded successfully.")

        st.write("Models and preprocessor loaded successfully!")
        return lstm_model, rnn_model, reg_model, gru_model, seq2seq_model, transformer_model, preprocessor
    except Exception as e:
        logging.error(f"Error loading models: {e}")
        st.error(f"Error loading models: {e}")
        st.stop()

@st.cache_resource
def load_vehicle_data():
    try:
        data = pd.read_csv('cleaned_vehicles_with_realistic_year_new.csv', low_memory=False)
        st.write("Vehicle data loaded successfully!")
        return data
    except Exception as e:
        st.error(f"Error loading vehicle data: {e}")
        raise

# Initialize models and data
lstm_model, rnn_model, reg_model, gru_model, seq2seq_model, transformer_model, preprocessor = load_models()
vehicle_data = load_vehicle_data()

# User Selection
st.sidebar.header("User Selection")
num_users = st.sidebar.number_input("Number of Users (1-5)", min_value=1, max_value=5, value=1, step=1)

user_selections = []

for i in range(num_users):
    st.sidebar.subheader(f"User {i + 1}")
    user_name = st.sidebar.text_input(f"Name of User {i + 1}", value=f"User {i + 1}")
    
    # Vehicle Selection
    user_make = st.sidebar.selectbox(f"Select Make for {user_name}", options=sorted(vehicle_data['make'].unique()), key=f"make_{i}")
    user_model = st.sidebar.selectbox(
        f"Select Model for {user_name}", 
        options=sorted(vehicle_data[vehicle_data['make'] == user_make]['model'].unique()), 
        key=f"model_{i}"
    )
    user_year = st.sidebar.selectbox(
        f"Select Year for {user_name}", 
        options=sorted(vehicle_data[(vehicle_data['make'] == user_make) & (vehicle_data['model'] == user_model)]['year'].unique(), reverse=True), 
        key=f"year_{i}"
    )
    user_engine_size = st.sidebar.number_input(f"Engine Size (L) for {user_name}", value=4.0, key=f"engine_{i}")
    user_cylinders = st.sidebar.number_input(f"Cylinders for {user_name}", value=4, step=1, key=f"cylinders_{i}")
    user_fuel_comb = st.sidebar.number_input(f"Fuel Consumption Comb (L/100 km) for {user_name}", value=13.0, key=f"fuel_{i}")

    user_vehicle = {
        "name": user_name,
        "make": user_make,
        "model": user_model,
        "year": user_year,
        "engine_size": user_engine_size,
        "cylinders": user_cylinders,
        "fuel_comb": user_fuel_comb
    }
    user_selections.append(user_vehicle)

# Display selected vehicles
st.subheader("Selected Vehicles")
for user in user_selections:
    st.write(f"**{user['name']}** selected:")
    st.write(f"- Make: {user['make']}")
    st.write(f"- Model: {user['model']}")
    st.write(f"- Year: {user['year']}")
    st.write(f"- Engine Size: {user['engine_size']} L")
    st.write(f"- Cylinders: {user['cylinders']}")
    st.write(f"- Fuel Consumption (Comb): {user['fuel_comb']} L/100 km")
    st.write("---")

# Comparison
if st.button("Compare Vehicles"):
    st.subheader("Vehicle Comparison Results")

    # Simple comparison based on fuel consumption
    comparison_df = pd.DataFrame(user_selections)
    comparison_df['CO₂ Emissions (kg/100 km)'] = comparison_df['fuel_comb'] * 2.3  # Example conversion factor

    # Display comparison table
    st.write("Comparison Table:")
    st.dataframe(comparison_df)

    # Highlight best vehicle (lowest emissions)
    best_vehicle = comparison_df.loc[comparison_df['CO₂ Emissions (kg/100 km)'].idxmin()]
    st.success(f"The best vehicle is selected by **{best_vehicle['name']}**, "
               f"with CO₂ emissions of {best_vehicle['CO₂ Emissions (kg/100 km)']:.2f} kg/100 km.")

# Page Title
st.title("CO₂ Emissions Prediction Dashboard")

# Sidebar Inputs for Vehicle Parameters

st.sidebar.header("Select Vehicle Parameters")
st.write("Rendering sidebar inputs...")
make = st.sidebar.selectbox("Select Make", options=sorted(vehicle_data['make'].unique()))
model = st.sidebar.selectbox("Select Model", options=sorted(vehicle_data[vehicle_data['make'] == make]['model'].unique()))
year = st.sidebar.selectbox("Select Year", options=sorted(vehicle_data[(vehicle_data['make'] == make) & (vehicle_data['model'] == model)]['year'].unique(), reverse=True))
engine_size = st.sidebar.number_input("Engine Size (L)", value=4.0)
cylinders = st.sidebar.number_input("Cylinders", value=4, step=1)
fuel_consumption_city = st.sidebar.number_input("Fuel Consumption City (L/100 km)", value=16.0)
fuel_consumption_hwy = st.sidebar.number_input("Fuel Consumption Hwy (L/100 km)", value=12.0)
fuel_consumption_comb = st.sidebar.number_input("Fuel Consumption Comb (L/100 km)", value=13.0)

# Additional Inputs for Missing Columns
vehicle_class = st.sidebar.selectbox("Vehicle Class (VClass)", options=["Compact", "Midsize", "SUV", "Truck", "Luxury"])
fuel_type = st.sidebar.selectbox("Fuel Type", options=["Gasoline", "Diesel", "Electric", "Hybrid"])
transmission = st.sidebar.selectbox("Transmission Type (trany)", options=["Automatic", "Manual", "CVT"])

st.write("Sidebar inputs collected successfully!")

# Prepare Input Data for Prediction
input_data = pd.DataFrame([[
    make, model, year, engine_size, cylinders, 
    fuel_consumption_city, fuel_consumption_hwy, fuel_consumption_comb, 
    vehicle_class, fuel_type, transmission
]],
    columns=['make', 'model', 'year', 'displ', 'cylinders', 
             'city08', 'highway08', 'comb08', 
             'VClass', 'fuelType', 'trany'])

st.write("Input data prepared:")
st.write(input_data)


# Prepare Input Data for Prediction
st.write("Preparing input data...")
input_data = pd.DataFrame([[
    make, model, year, engine_size, cylinders, 
    fuel_consumption_city, fuel_consumption_hwy, fuel_consumption_comb, 
    vehicle_class, fuel_type, transmission
]],
    columns=['make', 'model', 'year', 'displ', 'cylinders', 
             'city08', 'highway08', 'comb08', 
             'VClass', 'fuelType', 'trany'])

# Display input data after ensuring all columns are included
st.write("Input data prepared (with all required columns):")
st.write(input_data)

# Prediction Button with Descriptive Label
# Overlapping Line Graph for Model Predictions
if st.sidebar.button("Predict CO₂ Emissions", key="predict_button", help="Click to predict CO₂ emissions"):
    st.write("Prediction button clicked!")
    try:
        st.write("Preprocessing input data...")

        # Preprocess input data
        input_data_preprocessed = preprocessor.transform(input_data).toarray()
        lstm_input = input_data_preprocessed.reshape(
            (input_data_preprocessed.shape[0], 1, input_data_preprocessed.shape[1])
        )
        decoder_input = np.zeros((lstm_input.shape[0], 1, lstm_input.shape[2]))

        # Make predictions
        predictions = {
            "LSTM": lstm_model.predict(lstm_input)[0][0],
            "RNN": rnn_model.predict(lstm_input)[0][0],
            "Regression": reg_model.predict(input_data_preprocessed)[0][0],
            "GRU": gru_model.predict(lstm_input)[0][0],
            "Seq2Seq": seq2seq_model.predict([lstm_input, decoder_input])[0][0],
            "Transformer": transformer_model.predict(lstm_input)[0][0] if transformer_model else "N/A",
        }

        # Convert NumPy arrays to scalars
        valid_predictions = {k: (v.item() if isinstance(v, np.ndarray) else v) for k, v in predictions.items() if v != "N/A"}

        # Calculate carbon tax
        carbon_taxes = {model: calculate_carbon_tax(emission, carbon_tax_rate_per_ton) for model, emission in valid_predictions.items()}

        # Display predictions
        st.subheader("Prediction Results")
        for model, pred in valid_predictions.items():
            st.write(f"{model} Model Prediction: {pred:.2f} kg/100 km")
            st.write(f"{model} Carbon Tax: {carbon_taxes[model]:.2f} NT$/100 km")

        # Add a Bar Chart for Comparing Carbon Tax Across Models
        st.subheader("Carbon Tax Comparison Across Models")

        # Prepare data for visualization
        model_names = list(valid_predictions.keys())
        tax_values = [carbon_taxes[model] for model in model_names]

        # Create a bar chart using Matplotlib
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(model_names, tax_values, color='skyblue')

        # Customize the chart
        ax.set_xlabel("Models")
        ax.set_ylabel("Carbon Tax (NT$/100 km)")
        ax.set_title("Carbon Tax Comparison Across Models")
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Display the chart in Streamlit
        st.pyplot(fig)


        # Overlapping Line Graph
        st.subheader("Bar Chart of CO₂ Emissions by Model")
        fig, ax = plt.subplots(figsize=(8, 5))

        # Create a bar chart for the model predictions
        models = list(valid_predictions.keys())
        emissions = list(valid_predictions.values())

        # Plot the bar chart
        ax.bar(models, emissions, color='skyblue')

        # Customize the chart
        ax.set_xlabel("Models")
        ax.set_ylabel("CO₂ Emissions (kg/100 km)")
        ax.set_title("Bar Chart of Model Predictions")
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Display the chart
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error in prediction: {e}")

# Carbon Tax Calculator Section
st.subheader("CO₂ Emissions Tax Calculator")

# User Inputs for Tax Calculation
vehicle_type = st.selectbox("Select Vehicle Type", ["low-emission", "standard", "high-emission"])
emissions_per_km = st.number_input("Enter CO₂ Emissions (g/km)", value=150)
mileage = st.number_input("Enter Annual Mileage (km)", value=15000)
phase = st.radio("Select Implementation Phase", [1, 2], help="Phase 1: Lower tax rate (Years 1–3). Phase 2: Higher tax rate (Years 4–6).")

# Calculate Tax
if st.button("Calculate Tax"):
    try:
        # Adjust tax based on vehicle type or phased implementation
        if vehicle_type != "standard":
            annual_tax = adjust_tax_rate(emissions_per_km, mileage, vehicle_type)
        else:
            annual_tax = phased_implementation(emissions_per_km, mileage, phase)

        # Display Results
        st.success(f"Estimated Annual Tax: **{annual_tax:.2f} TWD**")
    except Exception as e:
        st.error(f"Error calculating tax: {e}")

# Cumulative Emissions Calculation (Updated Bar Chart)
st.subheader("Cumulative Emissions Calculator")
start_date = st.date_input("Start Date", datetime(2024, 1, 1))
end_date = st.date_input("End Date", datetime(2024, 12, 31))

if st.button("Calculate Cumulative Emissions", help="Calculate emissions over the selected date range"):
    try:
        days = (end_date - start_date).days + 1
        st.write(f"Calculating cumulative emissions for {days} days...")

        # Preprocess input data
        input_data_preprocessed = preprocessor.transform(input_data).toarray()
        lstm_input = input_data_preprocessed.reshape(
            (input_data_preprocessed.shape[0], 1, input_data_preprocessed.shape[1])
        )
        decoder_input = np.zeros((lstm_input.shape[0], 1, lstm_input.shape[2]))

        # Calculate cumulative emissions
        cumulative_predictions = {
            "LSTM": lstm_model.predict(lstm_input)[0][0] * days,
            "RNN": rnn_model.predict(lstm_input)[0][0] * days,
            "Regression": reg_model.predict(input_data_preprocessed)[0][0] * days,
            "GRU": gru_model.predict(lstm_input)[0][0] * days,
            "Seq2Seq": seq2seq_model.predict([lstm_input, decoder_input])[0][0] * days,
            "Transformer": transformer_model.predict(lstm_input)[0][0] * days if transformer_model else "N/A",
        }

        # Convert to scalars for proper formatting
        cumulative_predictions = {k: (v.item() if isinstance(v, np.ndarray) else v) for k, v in cumulative_predictions.items() if v != "N/A"}

        # Calculate cumulative carbon tax
        cumulative_carbon_taxes = {model: calculate_carbon_tax(emission, carbon_tax_rate_per_ton) for model, emission in cumulative_predictions.items()}

        # Display cumulative emissions and taxes
        st.subheader("Cumulative Emissions Results")
        for model, emission in cumulative_predictions.items():
            st.write(f"{model} Cumulative Emissions: {emission:.2f} kg")
            st.write(f"{model} Cumulative Carbon Tax: {cumulative_carbon_taxes[model]:.2f} NT$")

        # Bar Chart for Cumulative Emissions
        st.subheader("Bar Chart of Cumulative Emissions by Model")
        fig, ax = plt.subplots(figsize=(8, 5))

        models = list(cumulative_predictions.keys())
        emissions = list(cumulative_predictions.values())

        ax.bar(models, emissions, color='skyblue')
        ax.set_xlabel("Models")
        ax.set_ylabel("Cumulative CO₂ Emissions (kg)")
        ax.set_title("Bar Chart of Cumulative Emissions by Model")
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error in cumulative emissions calculation: {e}")

# Display Forecasted Performance Table with CO₂e Conversion
st.subheader("Forecasted Performance Table with CO₂e Conversion")

try:
    # Load the forecast performance data
    forecast_performance_file = "d:/MCU/Final Project/test061224/forecast_performance.csv"  # Replace with the actual path
    forecast_performance = pd.read_csv(forecast_performance_file)

    # Conversion factor (example: 1 kg CO₂ = 1.0 CO₂e, adjust as needed for specific factors)
    conversion_factor_co2e = 1.0  # Adjust based on your needs

    # Add a column for CO₂e
    forecast_performance["CO₂e (kg)"] = forecast_performance["MSE"] * conversion_factor_co2e

    # Display the table with the new column
    st.write("The following table shows the forecasted performance of various models with CO₂e conversion:")
    st.dataframe(forecast_performance)  # Use st.dataframe for scrollable tables or st.table for static ones

except Exception as e:
    st.error(f"Error displaying forecasted performance table: {e}")

# Prediction and Best Model Highlight (Updated Bar Chart)
st.write("Calculating predictions automatically...")

try:
    # Preprocess input data
    input_data_preprocessed = preprocessor.transform(input_data).toarray()
    lstm_input = input_data_preprocessed.reshape(
        (input_data_preprocessed.shape[0], 1, input_data_preprocessed.shape[1])
    )
    decoder_input = np.zeros((lstm_input.shape[0], 1, lstm_input.shape[2]))

    # Make predictions
    predictions = {
        "LSTM": lstm_model.predict(lstm_input)[0][0],
        "RNN": rnn_model.predict(lstm_input)[0][0],
        "Regression": reg_model.predict(input_data_preprocessed)[0][0],
        "GRU": gru_model.predict(lstm_input)[0][0],
        "Seq2Seq": seq2seq_model.predict([lstm_input, decoder_input])[0][0],
        "Transformer": transformer_model.predict(lstm_input)[0][0] if transformer_model else "N/A",
    }

    # Convert NumPy arrays to scalars
    valid_predictions = {k: (v.item() if isinstance(v, np.ndarray) else v) for k, v in predictions.items() if v != "N/A"}

    # Identify the "Best Model" (Lowest CO₂ Emission)
    best_model = min(valid_predictions, key=valid_predictions.get)
    best_emission = valid_predictions[best_model]
    best_carbon_tax = calculate_carbon_tax(best_emission, carbon_tax_rate_per_ton)

    # Display predictions
    st.subheader("Prediction Results")
    for model, pred in valid_predictions.items():
        st.write(f"{model} Model Prediction: {pred:.2f} kg/100 km")

    # Highlight Best Model
    st.success(f"The best model is **{best_model}**, predicting {best_emission:.2f} kg/100 km of CO₂ emissions.")
    st.success(f"The corresponding carbon tax is **{best_carbon_tax:.2f} NT$** per 100 km.")

    # Bar Chart for Model Predictions
    st.subheader("Bar Chart of Model Predictions")
    fig, ax = plt.subplots(figsize=(8, 5))

    models = list(valid_predictions.keys())
    emissions = list(valid_predictions.values())

    ax.bar(models, emissions, color='skyblue')
    ax.set_xlabel("Models")
    ax.set_ylabel("CO₂ Emissions (kg/100 km)")
    ax.set_title("Bar Chart of Model Predictions")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    st.pyplot(fig)

except Exception as e:
    st.error(f"Error in prediction: {e}")