# serve_model.py
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

import sys
import os

path= "C:\\Users\\Aman\\Desktop\\MODIFIED-FRAUD-DETECTION\\src"
sys.path.append(os.path.abspath(path=path))

from feature_engineering import featureEngineering

# Load the trained model
with open("C:\\Users\\Aman\\Desktop\\MODIFIED-FRAUD-DETECTION\\data\\preprocessed\\preprocessor.joblib", "rb") as file:
    preprocessor = joblib.load(file)

# Load scaler
with open("C:\\Users\\Aman\\Desktop\\MODIFIED-FRAUD-DETECTION\\data\\model\\mlps.joblib", "rb") as file:
    model = joblib.load(file)

# 'user_id','purchase_value', 'age', 'time_diff(hr)','day_of_week', 'ip_address','browser','source', 'country'
# features used in training
FEATURES = ['user_id', 'signup_time', 'purchase_time', 'purchase_value',
       'device_id', 'source', 'browser', 'sex','country', 'age', 'ip_address']

# Initialize Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", 'POST'])
def index():
    if request.method == "POST":
        try:
            # Get input values from form
            form_data = {feature: request.form[feature] for feature in FEATURES}
            # Convert datetime fields correctly
            form_data["signup_time"] = pd.to_datetime(form_data["signup_time"], errors="coerce")
            form_data["purchase_time"] = pd.to_datetime(form_data["purchase_time"], errors="coerce")

            # Check if conversion was successful
            if pd.isnull(form_data["signup_time"]) or pd.isnull(form_data["purchase_time"]):
                return jsonify({"error": "Invalid date format. Please use YYYY-MM-DD HH:MM:SS"}), 400

            # Convert to DataFrame
            df = pd.DataFrame([form_data])

            # Ensure numeric columns are converted properly
            numeric_features = ['user_id',  'purchase_value', 'age', 'ip_address']
            df[numeric_features] = df[numeric_features].astype(float)

            feature = featureEngineering(df)
            featured_data = feature.feature_extraction()

            numerical_columns =['user_id','purchase_value', 'age', 'time_diff(hr)','day_of_week', 'ip_address']
            categorical_colmns =['browser','source']

            processed = preprocessor.transform(featured_data)

            # Make prediction
            prediction = model.predict(processed)[0]

            # Return result
            return render_template("index.html", prediction=prediction)  # Ensure it's an integer


        except Exception as e:
            return jsonify({"error": str(e)})

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         data = request.get_json()  # Get input data in JSON format
#         features = np.array(data["features"]).reshape(1, -1)  # Convert to numpy array
#         prediction = model.predict(features)  # Predict
#         return jsonify({"prediction": int(prediction[0])})  # Return JSON response
#     except Exception as e:
#         return jsonify({"error": str(e)}), 400

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)



