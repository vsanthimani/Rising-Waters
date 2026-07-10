from flask import Flask, render_template_string, request

app = Flask(__name__)

# ==========================================
# 1. SHARED STATIC ASSETS (CSS & JAVASCRIPT)
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
    background-color: var(--bg-color);
    color: var(--text-color);
    margin: 0; padding: 0;
    display: flex; justify-content: center; align-items: center;
    min-height: 100vh;
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
    text-decoration: none; font-weight: bold; cursor: pointer;
    transition: background 0.2s; border: none;
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
.metrics { background: #edf2f7; padding: 15px; border-radius: 6px; text-align: left; margin: 20px 0;}
.metrics ul { list-style: none; padding: 0; margin: 0;}
.metrics li { margin: 8px 0; }
.button-group * { margin: 5px; }
"""

FORM_VALIDATION_JS = """
document.addEventListener("DOMContentLoaded", function() {
    const form = document.getElementById("predictionForm");
    if (form) {
        form.addEventListener("submit", function(event) {
            const rainfall = parseFloat(document.getElementById("rainfall").value);
            const humidity = parseFloat(document.getElementById("humidity").value);
            const riverLevel = parseFloat(document.getElementById("river_level").value);

            if (rainfall < 0 || humidity < 0 || riverLevel < 0) {
                alert("Weather parameters cannot be negative values.");
                event.preventDefault();
                return;
            }
            if (humidity > 100) {
                alert("Humidity cannot exceed 100%.");
                event.preventDefault();
                return;
            }
            const submitBtn = form.querySelector(".btn-submit");
            submitBtn.innerText = "Processing Metrics...";
            submitBtn.style.opacity = "0.7";
        });
    }
});
"""

# ==========================================
# 2. HTML TEMPLATES
# ==========================================

HOME_HTML = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Flood Prediction System - Home</title>
    <style>{SHARED_CSS}</style>
</head>
<body>
    <div class="container">
        <h1>Flood Prediction System</h1>
        <p class="subtitle">Utilizing advanced weather parameters to keep communities safe.</p>
        <div class="card">
            <h3>About the Project</h3>
            <p>This application analyzes crucial meteorological indicators such as rainfall depth, humidity levels, and river overflow thresholds to accurately forecast potential flood risks before they happen.</p>
        </div>
        <div class="action-section">
            <a href="/predict" class="btn btn-primary">Launch Predictor Tool</a>
        </div>
    </div>
</body>
</html>
"""

INDEX_HTML = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Weather Input Form</title>
    <style>{SHARED_CSS}</style>
</head>
<body>
    <div class="container">
        <a href="/" class="btn-back">← Back to Home</a>
        <h2>Flood Risk Analyzer</h2>
        <p>Please enter the current weather and hydrological observations below.</p>
        
        <form id="predictionForm" action="/predict" method="POST" class="card">
            <div class="form-group">
                <label for="rainfall">Rainfall (mm):</label>
                <input type="number" step="0.01" id="rainfall" name="rainfall" required placeholder="e.g., 120.5">
            </div>
            <div class="form-group">
                <label for="humidity">Humidity (%):</label>
                <input type="number" step="0.01" id="humidity" name="humidity" required placeholder="e.g., 85">
            </div>
            <div class="form-group">
                <label for="river_level">River Level (m):</label>
                <input type="number" step="0.01" id="river_level" name="river_level" required placeholder="e.g., 3.2">
            </div>
            <button type="submit" class="btn btn-submit">Analyze Data</button>
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
    <meta charset="UTF-8">
    <title>Flood Alert - High Risk</title>
    <style>{SHARED_CSS}</style>
</head>
<body class="bg-alert">
    <div class="container text-center">
        <div class="status-card alert-card">
            <span class="icon">⚠️</span>
            <h1>High Flood Risk Detected</h1>
            <p>Based on the weather parameters provided, there is a strong probability of imminent flooding in the area.</p>
            
            <div class="metrics">
                <h4>Submitted Observations:</h4>
                <ul>
                    <li>Rainfall: <strong>{{{{ rainfall }}}} mm</strong></li>
                    <li>Humidity: <strong>{{{{ humidity }}}}%</strong></li>
                    <li>River Level: <strong>{{{{ river_level }}}} m</strong></li>
                </ul>
            </div>
            
            <div class="button-group">
                <a href="/predict" class="btn btn-secondary">Run Another Check</a>
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
    <meta charset="UTF-8">
    <title>Flood Prediction - Safe</title>
    <style>{SHARED_CSS}</style>
</head>
<body class="bg-safe">
    <div class="container text-center">
        <div class="status-card safe-card">
            <span class="icon">✅</span>
            <h1>No Immediate Flood Risk</h1>
            <p>The current meteorological conditions suggest conditions are stable. No active flood alerts are triggered for these parameters.</p>
            
            <div class="button-group">
                <a href="/predict" class="btn btn-primary">Run Another Check</a>
                <a href="/" class="btn btn-outline">Home</a>
            </div>
        </div>
    </div>
</body>
</html>
"""

# ==========================================
# 3. BACKEND ROUTE CONTROLLERS
# ==========================================

@app.route('/')
def home():
    return render_template_string(HOME_HTML)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            rainfall = float(request.form.get('rainfall', 0))
            humidity = float(request.form.get('humidity', 0))
            river_level = float(request.form.get('river_level', 0))
            
            # --- PLACEHOLDER PREDICTION LOGIC ---
            # Triggers a flood alert if rainfall > 250mm OR river > 5m when rainfall is over 100mm
            if rainfall > 250 or (river_level > 5 and rainfall > 100):
                return render_template_string(CHANCE_HTML, rainfall=rainfall, humidity=humidity, river_level=river_level)
            else:
                return render_template_string(NO_CHANCE_HTML)
                
        except ValueError:
            return "Invalid input data. Please enter numbers only.", 400
            
    return render_template_string(INDEX_HTML)

if __name__ == '__main__':
    app.run(debug=True)