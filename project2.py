from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


@app.route('/')
def home():
    return render_template("home.html")


@app.route('/predict')
def predict():
    return render_template("index.html")


@app.route('/result', methods=['POST'])
def result():
    try:
        rainfall = float(request.form['rainfall'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])

        # Create input array
        data = np.array([[rainfall, temperature, humidity]])

        # Scale the input
        scaled_data = scaler.transform(data)

        # Predict
        prediction = model.predict(scaled_data)

        if prediction[0] == 1:
            return render_template("chance.html")
        else:
            return render_template("no_chance.html")

    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    app.run(debug=True)