

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("All required libraries imported successfully!")

data = {
    "Annual_Rainfall": [1200, 900, 1500, 800, 1100],
    "Temperature": [30, 35, 28, 36, 29],
    "Humidity": [85, 70, 92, 65, 88],
    "Flood": [1, 0, 1, 0, 1]
}


df = pd.DataFrame(data)

print("\nDataset")
print(df)


X = df[["Annual_Rainfall", "Temperature", "Humidity"]]
y = df["Flood"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


prediction = model.predict(X_test)


accuracy = accuracy_score(y_test, prediction)

print("\nModel Accuracy:", accuracy)


plt.figure(figsize=(6,4))
sns.barplot(x=["Flood","No Flood"], y=[3,2])

plt.title("Flood Prediction Sample Data")
plt.show()

app = Flask(__name__)

@app.route("/")
def home():
    return "Flood Prediction System Running Successfully!"

print("\nFlask Application Created Successfully!")