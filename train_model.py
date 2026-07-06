# train_model.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# =====================================================================
# 1. GENERATE OR LOAD DATA (Using NumPy & Pandas)
# =====================================================================
print("--- Step 1: Generating Synthetic Data ---")
np.random.seed(42)

# Simulating a dataset: 1000 rows, 4 features (Age, Income, Score, Debt)
n_samples = 1000
data = {
    'Age': np.random.randint(18, 70, size=n_samples),
    'Income': np.random.normal(50000, 15000, size=n_samples),
    'CreditScore': np.random.randint(300, 850, size=n_samples),
    'Debt': np.random.uniform(0, 20000, size=n_samples)
}

# Turn it into a Pandas DataFrame
df = pd.DataFrame(data)

# Create a dummy target variable (1 = Churn/Risk, 0 = Stay/Safe) based on a simple logic
# (e.g., if debt is high and credit score is low, risk is higher)
df['Target'] = ((df['Debt'] / df['Income'] > 0.3) & (df['CreditScore'] < 600)).astype(int)

print(df.head())
print(f"Target distribution:\n{df['Target'].value_counts()}\n")

# =====================================================================
# 2. DATA PREPROCESSING & SPLITTING (Using Scikit-learn)
# =====================================================================
print("--- Step 2: Preprocessing and Splitting ---")
X = df.drop(columns=['Target'])
y = df['Target']

# Split data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Scaling (Crucial for many models, good practice overall)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================================
# 3. MODEL TRAINING & EVALUATION (Using Scikit-learn)
# =====================================================================
print("--- Step 3: Training Random Forest Classifier ---")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Print evaluation metrics
print("\nModel Evaluation Report:")
print(classification_report(y_test, y_pred))

# =====================================================================
# 4. DATA VISUALIZATION (Using Matplotlib & Seaborn)
# =====================================================================
print("--- Step 4: Generating and Saving Visualizations ---")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Safe', 'Risk'], yticklabels=['Safe', 'Risk'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')

# Save the plot as an image file
plt.savefig('confusion_matrix.png')
print("Saved confusion matrix visualization to 'confusion_matrix.png'")
plt.close()

# =====================================================================
# 5. EXPORT MODEL & SCALER (Using Pickle)
# =====================================================================
print("\n--- Step 5: Exporting Model Assets ---")
# Create an assets directory if it doesn't exist
os.makedirs('models', exist_ok=True)

with open('models/random_forest_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Successfully exported model and scaler to the 'models/' directory!")