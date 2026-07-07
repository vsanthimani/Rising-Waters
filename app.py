# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# print("All libraries imported successfully!")

# ==========================================
# FLOOD PREDICTION MODEL TRAINING
# ==========================================

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

from xgboost import XGBClassifier
from gradient_boosting_model import gradient_boosting

import joblib

# ==========================================
# LOAD DATASET
# ==========================================

print("Loading Dataset...")

dataset = pd.read_excel("../dataset/flood_dataset.xlsx")

print("\nFirst 5 Rows:")
print(dataset.head())

# ==========================================
# DATA EXPLORATION
# ==========================================

print("\nDataset Shape:")
print(dataset.shape)

print("\nDataset Information:")
print(dataset.info())

print("\nStatistical Summary:")
print(dataset.describe())

print("\nMissing Values:")
print(dataset.isnull().sum())

# ==========================================
# HANDLE MISSING VALUES
# ==========================================

dataset.fillna(dataset.mean(numeric_only=True), inplace=True)

# ==========================================
# FEATURES AND TARGET
# ==========================================

X = dataset[
    [
        "annual_rainfall",
        "cloud_visibility",
        "seasonal_rainfall",
        "temperature",
        "humidity"
    ]
]

y = dataset["flood"]

# ==========================================
# SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================================
# FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================
# TRAIN MODEL
# ==========================================

print("\nTraining Model...")

model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================================
# MODEL TRAINING
# ==========================================

model, y_pred = gradient_boosting(
    X_train,
    X_test,
    y_train,
    y_test
)

# ==========================================
# PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(round(accuracy * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(model, "flood_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel Saved Successfully!")

# ==========================================
# SAMPLE PREDICTION
# ==========================================

sample_data = pd.DataFrame({
    "annual_rainfall": [1250.5],
    "cloud_visibility": [70],
    "seasonal_rainfall": [450.2],
    "temperature": [29.5],
    "humidity": [82]
})

sample_data = scaler.transform(sample_data)

prediction = model.predict(sample_data)

print("\nPrediction Result:")

if prediction[0] == 1:
    print("Flood Likely")
else:
    print("No Flood")

# ==========================================
# VISUALIZATION
# ==========================================

plt.figure(figsize=(8, 5))
sns.heatmap(dataset.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# ==========================================
# UNIVARIATE ANALYSIS
# ==========================================

import matplotlib.pyplot as plt
import seaborn as sns

# Distribution Plot - Temperature

plt.figure(figsize=(6,4))
sns.histplot(dataset['temperature'], kde=True)
plt.title("Temperature Distribution")
plt.show()

# Box Plot - Temperature

plt.figure(figsize=(6,4))
sns.boxplot(x=dataset['temperature'])
plt.title("Temperature Box Plot")
plt.show()

# ==========================================
# DISTRIBUTION PLOTS
# ==========================================

columns = [
    'annual_rainfall',
    'cloud_visibility',
    'seasonal_rainfall',
    'temperature',
    'humidity'
]

for col in columns:
    plt.figure(figsize=(6,4))
    sns.histplot(dataset[col], kde=True)
    plt.title(f"Distribution Plot - {col}")
    plt.show()

# ==========================================
# BOX PLOTS
# ==========================================

for col in columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=dataset[col])
    plt.title(f"Box Plot - {col}")
    plt.show()


# ==========================================
# MULTIVARIATE ANALYSIS
# CORRELATION HEATMAP
# ==========================================

plt.figure(figsize=(12,8))

sns.heatmap(
    dataset.corr(),
    annot=True,
    cmap='summer',
    linewidths=1,
    linecolor='black',
    square=True,
    cbar=True
)

plt.title("Correlation Heatmap")
plt.show()