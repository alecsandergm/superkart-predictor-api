# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API
import os

# Initialize the Flask application
superkart_predictor_api = Flask("SuperKart Predictor Sales")

# Load the trained machine learning model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR,
    "superkart_prediction_model_v1_0.joblib"
)
model = joblib.load(MODEL_PATH)

# Define a route for the home page (GET request)
@superkart_predictor_api.get('/')
def home():
    """
    This function handles GET requests to the root URL ('/') of the API.
    It returns a simple welcome message.
    """
    return "Welcome to the SuperKart Product Sales Prediction API!"

# Define an endpoint for single product prediction (POST request)
@superkart_predictor_api.post('/v1/sales')
def predict_product_sales_function():
    """
    This function handles POST requests to the '/v1/sales' endpoint.
    It expects a JSON payload containing property details and returns
    the predicted product sales as a JSON response.
    """
    # Get the JSON data from the request body
    product_data = request.get_json()

    # Extract relevant features from the JSON data
    sample = {
        'Product_Weight': product_data['Product_Weight'],
        'Product_Sugar_Content': product_data['Product_Sugar_Content'],
        'Product_Allocated_Area': product_data['Product_Allocated_Area'],
        'Product_Type': product_data['Product_Type'],
        'Product_MRP': product_data['Product_MRP'],
        'Store_Id': product_data['Store_Id'],
        'Store_Establishment_Year': product_data['Store_Establishment_Year'],
        'Store_Size': product_data['Store_Size'],
        'Store_Location_City_Type': product_data['Store_Location_City_Type'],
        'Store_Type': product_data['Store_Type']
    }

    # Convert the extracted data into a Pandas DataFrame
    input_data = pd.DataFrame([sample])

    # Make prediction
    predicted_product_sale = model.predict(input_data)[0]

    # Convert prediction back to original scale
    predicted_sale = predicted_product_sale

    # Convert NumPy value to Python float
    predicted_sale = round(float(predicted_sale), 2)

    # Return prediction
    return jsonify({'Predicted Product Sold': predicted_sale})


# Define an endpoint for batch prediction (POST request)
@superkart_predictor_api.post('/v1/salesbatch')
def predict_product_sales_batch_function():
    """
    Handles POST requests to the '/v1/salesbatch' endpoint.
    Expects a CSV file containing multiple products
    and returns predicted product sales as a JSON response.
    """

    # Get the uploaded CSV file
    file = request.files['file']

    # Read the CSV file into a Pandas DataFrame
    input_data = pd.read_csv(file)

    # Select only the features used by the model
    model_input = input_data[numeric_features + categorical_features]

    # Make predictions
    predicted_product_sales = model.predict(model_input).tolist()

    # Round predictions
    predicted_sales = [
        round(float(predicted_product_sale), 2)
        for predicted_product_sale in predicted_product_sales
    ]

    # Return predictions as a JSON response
    return jsonify(predicted_sales)

# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    rental_price_predictor_api.run(debug=True)    
