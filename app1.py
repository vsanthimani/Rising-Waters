from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Global Variable Containers for Pipeline Artifacts
model = None
scaler = None
feature_columns = []

def load_artifacts():
    global model, scaler, feature_columns
    try:
        with open("flood_model.pkl", "rb") as m_file:
            model = pickle.load(m_file)
        with open("scaler.pkl", "rb") as s_file:
            scaler = pickle.load(s_file)
        with open("features.pkl", "rb") as f_file:
            feature_columns = pickle.load(f_file)
        print("[✓] Machine Learning artifacts loaded successfully.")
        return True
    except Exception as e:
        print(f"[!] Warning: Artifacts loading failed: {e}. Run 'train_model.py' first.")
        return False

@app.route('/', methods=['GET'])
def index():
    # Render the input landing view UI template page mapping existing feature columns
    return render_template('index.html', features=feature_columns)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Model files are missing on host engine application environment."}), 500
    
    try:
        # Extract features dynamically from the form interface based on baseline schema arrays
        input_data = []
        for feature in feature_columns:
            val = request.form.get(feature)
            if val is None:
                return f"Missing value component parameter for field feature: {feature}", 400
            input_data.append(float(val))
            
        # Convert into standard 2D shape format array structure matching input matrix expectation
        final_features = [np.array(input_data)]
        
        # Apply extracted scaling parameters
        scaled_features = scaler.transform(final_features)
        
        # Generate prediction
        prediction_output = model.predict(scaled_features)[0]
        
        # Format display boundary protections limits
        probability_percentage = round(max(0.0, min(1.0, float(prediction_output))) * 100, 2)
        
        return render_template('index.html', 
                               features=feature_columns, 
                               prediction_text=f'Calculated Regional Flood Occurrence Probability: {probability_percentage}%')
                               
    except Exception as e:
        return render_template('index.html', 
                               features=feature_columns, 
                               prediction_text=f'Error Processing Input Request Parameters: {str(e)}')

if __name__ == '__main__':
    # Initialize dynamic models structural layers mapping verification
    load_artifacts()
    # Execute Local Host Execution Loop
    app.run(host='0.0.0.0', port=5000, debug=True)