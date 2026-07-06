# app.py
import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# =====================================================================
# LOAD MODEL ASSETS ON STARTUP
# =====================================================================
try:
    with open('models/random_forest_model.pkl', 'rb') as f:
        model = pickle.pickle.load(f) if hasattr(pickle, 'pickle') else pickle.load(f)
        
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("Model assets loaded successfully. Ready for predictions!")
except FileNotFoundError:
    print("Error: Model files not found! Please run 'train_model.py' first to generate them.")
    exit(1)

# =====================================================================
# API ROUTES
# =====================================================================

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Welcome to the ML Model Deployment API!",
        "status": "Running",
        "usage": "Send a POST request to /predict with JSON payload containing 'Age', 'Income', 'CreditScore', and 'Debt'."
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Parse JSON data from incoming request
        data = request.get_json(force=True)
        
        # Expected input format example:
        # { "Age": 34, "Income": 65000, "CreditScore": 710, "Debt": 5000 }
        
        # 2. Extract features in correct order
        features = [
            data['Age'],
            data['Income'],
            data['CreditScore'],
            data['Debt']
        ]
        
        # 3. Convert to 2D numpy array for the model
        features_array = np.array([features])
        
        # 4. Apply the same scaling used during training
        features_scaled = scaler.transform(features_array)
        
        # 5. Perform prediction and extract probabilities
        prediction = int(model.predict(features_scaled)[0])
        probability = model.predict_proba(features_scaled)[0]
        risk_probability = float(probability[1])
        
        # Map target back to readable label
        status_label = "Risk / Churn" if prediction == 1 else "Safe / Retained"
        
        # 6. Return response
        return jsonify({
            "status": "success",
            "prediction": prediction,
            "label": status_label,
            "risk_probability": round(risk_probability, 4)
        })

    except KeyError as e:
        return jsonify({"status": "error", "message": f"Missing expected feature: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# =====================================================================
# RUN THE FLASK APP
# =====================================================================
if __name__ == '__main__':
    # Run locally on http://127.0.0.1:5000/
    app.run(debug=True, port=5000)