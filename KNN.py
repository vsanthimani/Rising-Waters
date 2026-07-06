# ==========================================
# Flood Prediction using K-Nearest Neighbors
# ==========================================

# Import Required Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ==========================================
# KNN Function
# ==========================================

def KNN(X_train, X_test, y_train, y_test):

    print("==========================================")
    print("      K-Nearest Neighbors (KNN)")
    print("==========================================")

    # Initialize Model
    model = KNeighborsClassifier(n_neighbors=5)

    # Train Model
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    # Classification Report
    report = classification_report(y_test, y_pred)

    # Print Results
    print("\nAccuracy Score")
    print("---------------------")
    print(round(accuracy * 100, 2), "%")

    print("\nConfusion Matrix")
    print("---------------------")
    print(cm)

    print("\nClassification Report")
    print("---------------------")
    print(report)

    return model, y_pred


# ==========================================
# Main Program
# ==========================================

# Load Dataset
df = pd.read_csv("dataset.csv")

print("\nDataset Loaded Successfully!\n")

print(df.head())

# Input Features
X = df.drop("Flood", axis=1)

# Target Variable
y = df["Flood"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Call Function
model, predictions = KNN(
    X_train,
    X_test,
    y_train,
    y_test
)

print("\nModel Training Completed Successfully!")