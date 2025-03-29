from flask import Flask, request, jsonify
import joblib
import numpy as np
import xgboost as xgb
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("pm25_xgboost_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

        # Perform prediction
        predicted_pm25 = model.predict(input_df)[0]

        # Return result as JSON
        return jsonify({'predicted_pm25': round(float(predicted_pm25), 2)})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
