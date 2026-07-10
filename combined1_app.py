from flask import Flask, render_template_string, request, redirect, url_for
import joblib
import pandas as pd
import numpy as np

# ==========================================
# 1. SETUP AND INITIALIZATION
# ==========================================
app = Flask(__name__)

# Load the saved model and scaler assets
try:
    model = joblib.load('floods.save')
    scaler = joblib.load('transform.save')
    print("Model and Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

# ==========================================
# 2. SHARED STATIC ASSETS (CSS & JS)
# ==========================================
SHARED_CSS = """
:root {
    --primary-color: #2b6cb0;
    --danger-color: #e53e3e;
    --success-color: #38a169;
    --bg-color: #f7fafc;
    --text-color: #2d3748;
}
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: var(--bg-color); color: var(--text-color);
    margin: 0; padding: 0; display: flex; justify-content: center; align-items: center; min-height: 100vh;
}
.container { width: 90%; max-width: 600px; margin: 20px auto; }
.text-center { text-align: center; }
h1, h2, h3 { color: #1a202c; }
.subtitle { color: #718096; font-size: 1.1rem; margin-bottom: 2rem;}
.card {
    background: #ffffff; padding: 30px; border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px;
}
.form-group { margin-bottom: 20px; }
label { display: block; margin-bottom: 8px; font-weight: 600; }
input[type="number"] {
    width: 100%; padding: 10px; border: 1px solid #cbd5e0;
    border-radius: 4px; box-sizing: border-box; font-size: 1rem;
}
.btn {
    display: inline-block; padding: 12px 24px; border-radius: 4px;
    text-decoration: none; font-weight: bold; cursor: pointer; border: none;
}
.btn-primary { background: var(--primary-color); color: white; }
.btn-submit { background: var(--primary-color); color: white; width: 100%; font-size: 1rem; }
.btn-secondary { background: #718096; color: white; }
.btn-outline { border: 1px solid #cbd5e0; color: var(--text-color); }
.btn-back { display: inline-block; margin-bottom: 15px; color: var(--primary-color); text-decoration: none; }
.bg-alert { background-color: #fff5f5; }
.bg-safe { background-color: #f0fff4; }
.status-card { padding: 40px; border-radius: 12px; background: white; box-shadow: 0 10px 15px rgba(0,0,0,0.05); }
.alert-card { border-top: 8px solid var(--danger-color); }
.safe-card { border-top: 8px solid var(--success-color); }
.icon { font-size: 3.5rem; display: block; margin-bottom: 15px; }
.button-group * { margin: 5px; }
"""

FORM_VALIDATION_JS = """
document.addEventListener("DOMContentLoaded", function() {
    const form = document.getElementById("predictionForm");
    if (form) {
        form.addEventListener("submit", function(event) {
            const inputs = form.querySelectorAll("input[type='number']");
            for (let input of inputs) {
                if (parseFloat(input.value) < 0) {
                    alert("Weather parameters cannot be negative values.");
                    event.preventDefault();
                    return;
                }
            }
            const submitBtn = form.querySelector(".btn-submit");
            submitBtn.innerText = "Processing Parameters...";
            submitBtn.style.opacity = "0.7";
        });
    }
});
"""

# ==========================================
# 3. HTML EMBEDDED VIEWS
# ==========================================
HOME_HTML = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"><title>Flood Prediction System - Home</title>
    <style>{SHARED_CSS}</style>
</head>
<body>
    <div class="container">
        <h1>Flood Prediction System</h1>
        <p class="subtitle">Utilizing advanced weather parameters to keep communities safe.</p>
        <div class="card">
            <h3>About the Project</h3>
            <p>This system utilizes an optimized Machine Learning pipeline to analyze distinct hydrological and meteorological factors to determine flood vulnerabilities in real time.</p>
        </div>
        <div class="action-section">
            <a href="/Predict" class="btn btn-primary">Launch Predictor Tool</a>
        </div>
    </div>
</body>
</html>
"""

INDEX_HTML = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"><title>Weather Input Form</title>
    <style>{SHARED_CSS}</style>
</head>
<body>
    <div class="container">
        <a href="/" class="btn-back">← Back to Home</a>
        <h2>Flood Risk Analyzer</h2>
        <p>Please provide the 5 predictive environmental metrics required by the model:</p>
        
        <form id="predictionForm" action="/Predict" method="POST" class="card">
            <div class="form-group">
                <label for="feat1">Parameter 1 (e.g., Rainfall mm):</label>
                <input type="number" step="any" id="feat1" name="feature1" required>
            </div>
            <div class="form-group">
                <label for="feat2">Parameter 2 (e.g., Humidity %):</label>
                <input type="number" step="any" id="feat2" name="feature2" required>
            </div>
            <div class="form-group">
                <label for="feat3">Parameter 3 (e.g., River Stage m):</label>
                <input type="number" step="any" id="feat3" name="feature3" required>
            </div>
            <div class="form-group">
                <label for="feat4">Parameter 4 (e.g., Soil Moisture):</label>
                <input type="number" step="any" id="feat4" name="feature4" required>
            </div>
            <div class="form-group">
                <label for="feat5">Parameter 5 (e.g., Temperature °C):</label>
                <input type="number" step="any" id="feat5" name="feature5" required>
            </div>
            <button type="submit" class="btn btn-submit">Analyze Matrix</button>
        </form>
    </div>
    <script>{FORM_VALIDATION_JS}</script>
</body>
</html>
"""

CHANCE_HTML = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"><title>Flood Alert - High Risk</title>
    <style>{SHARED_CSS}</style>
</head>
<body class="bg-alert">
    <div class="container text-center">
        <div class="status-card alert-card">
            <span class="icon">⚠️</span>
            <h1>High Flood Risk Detected</h1>
            <p>The trained predictive model indicates an imminent threat of flooding given the parsed environmental characteristics.</p>
            <div class="button-group">
                <a href="/Predict" class="btn btn-secondary">Analyze New Data</a>
                <a href="/" class="btn btn-outline">Home</a>
            </div>
        </div>
    </div>
</body>
</html>
"""

NO_CHANCE_HTML = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"><title>Flood Prediction - Safe</title>
    <style>{SHARED_CSS}</style>
</head>
<body class="bg-safe">
    <div class="container text-center">
        <div class="status-card safe-card">
            <span class="icon">✅</span>
            <h1>No Immediate Flood Risk</h1>
            <p>Model analysis complete. The evaluated atmospheric and hydrological profiles remain within stable metrics.</p>
            <div class="button-group">
                <a href="/Predict" class="btn btn-primary">Analyze New Data</a>
                <a href="/" class="btn btn-outline">Home</a>
            </div>
        </div>
    </div>
</body>
</html>
"""

# ==========================================
# 4. PAGE ROUTING & PREDICTION LOGIC
# ==========================================

# / routes to home.html (Landing page)
@app.route('/')
def home():
    return render_template_string(HOME_HTML)

# /Predict routes to index.html (Input form)
@app.route('/Predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if model is None or scaler is None:
            return "Error: Machine Learning files ('floods.save' or 'transform.save') missing or corrupt on server.", 500
        
        try:
            # Gather the 5 independent parameters
            f1 = float(request.form.get('feature1', 0))
            f2 = float(request.form.get('feature2', 0))
            f3 = float(request.form.get('feature3', 0))
            f4 = float(request.form.get('feature4', 0))
            f5 = float(request.form.get('feature5', 0))
            
            # Map parameters into a 5-feature DataFrame structured exactly like the training setup
            # NOTE: Alter columns list ['Col1', ...] to precisely match your original training dataset columns
            input_df = pd.DataFrame([[f1, f2, f3, f4, f5]], columns=['Col1', 'Col2', 'Col3', 'Col4', 'Col5'])
            
            # Normalize and scale metrics
            scaled_input = scaler.transform(input_df)
            
            # Generate classification array
            prediction = model.predict(scaled_input)
            
            # Conditional assessment route switching
            if prediction[0] == 1:
                return redirect(url_for('chance'))
            else:
                return redirect(url_for('no_chance'))
                
        except ValueError:
            return "Bad Request: Ensure all 5 numeric inputs are configured correctly.", 400
        except Exception as e:
            return f"Processing Error: {str(e)}", 500
            
    return render_template_string(INDEX_HTML)

# /chance routes to chance.html (Flood predicted)
@app.route('/chance')
def chance():
    return render_template_string(CHANCE_HTML)

# /no_chance routes to no_chance.html (No flood predicted)
@app.route('/no_chance')
def no_chance():
    return render_template_string(NO_CHANCE_HTML)

# ==========================================
# 5. EXECUTION BLOCK
# ==========================================
if __name__ == '__main__':
    # Launches the localized dev server environment on localhost:5000
    app.run(debug=True)