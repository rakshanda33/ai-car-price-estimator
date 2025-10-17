# Import necessary libraries
from flask import Flask, request, jsonify
from flask_cors import CORS  # <-- 1. ADDED THIS IMPORT
import joblib
import pandas as pd
import numpy as np

# Create a Flask application
app = Flask(__name__)
CORS(app)  # <-- 2. ADDED THIS LINE TO ENABLE CROSS-ORIGIN REQUESTS

# --- Load All the Saved Files ---
# Load the trained Random Forest regression model
try:
    model = joblib.load('car_price_predictor_model.joblib')
    print("Regression model loaded successfully.")
except Exception as e:
    print(f"Error loading regression model: {e}")
    model = None

# Load the model columns
try:
    model_columns = joblib.load('model_columns.joblib')
    print("Model columns loaded successfully.")
except Exception as e:
    print(f"Error loading model columns: {e}")
    model_columns = None

# --- Define the API Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    # Check if model and columns were loaded correctly
    if model is None or model_columns is None:
        return jsonify({'error': 'Model or supporting files are not loaded properly. Check server logs.'}), 500

    try:
        # Get the JSON data sent from the frontend
        json_data = request.get_json()
        
        # Create a pandas DataFrame from the single data point
        input_df = pd.DataFrame([json_data])
        
        # Create a new DataFrame with the same columns as the training data, filled with zeros
        # This ensures the input for the model has the exact same structure
        final_df = pd.DataFrame(columns=model_columns)
        final_df.loc[0] = 0

        # --- One-Hot Encode the Input Data ---
        # Populate the final_df with the user's input
        for col in input_df.columns:
            value = input_df[col].iloc[0]
            
            # For categorical features, find the matching one-hot encoded column and set it to 1
            if isinstance(value, str):
                column_name = f"{col}_{value}"
                if column_name in final_df.columns:
                    final_df.loc[0, column_name] = 1
            # For numerical features, just set the value directly
            else:
                if col in final_df.columns:
                    final_df.loc[0, col] = value
        
        # Make the prediction
        prediction = model.predict(final_df)
        
        # Get the first prediction and convert it to a standard Python float
        output = float(prediction[0])
        
        # Return the prediction as a JSON response
        return jsonify({'predicted_price': output})
        
    except Exception as e:
        # Return a detailed error if something goes wrong during prediction
        print(f"Error during prediction: {e}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

# --- Run the Flask App ---
if __name__ == '__main__':
    # Start the server, making it accessible on your local network
    app.run(port=5000, debug=True)

