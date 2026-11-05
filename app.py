# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
 
# Load the saved model when the server starts
model = joblib.load("iris_model.pkl")
 
# Create the Flask web application
app = Flask(__name__)
 
 
# Route 1: Home page — just confirms the server is running
@app.route("/")
def home():
    return "Iris Classifier API is Running!"
 
 
# Route 2: Prediction — receives 4 numbers, returns species name
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read the JSON data from the incoming request
        data = request.get_json(force=True)
 
        # Check that exactly 4 numbers were sent
        if "features" not in data or len(data["features"]) != 4:
            return jsonify({"error": "Exactly 4 numerical features are required"}), 400
 
        # Convert to a NumPy array (the format the model expects)
        features = np.array(data["features"], dtype=float).reshape(1, -1)
 
        # Run the prediction (returns 0, 1 or 2)
        prediction = model.predict(features)[0]
 
        # Convert number to species name
        classes = ["setosa", "versicolor", "virginica"]
        return jsonify({"prediction": classes[prediction]})
 
    except Exception as e:
        return jsonify({"error": str(e)}), 400
 
 
if __name__ == "__main__":
    # host=0.0.0.0 makes it accessible from outside (needed for Render)
    app.run(host="0.0.0.0", port=5000)
