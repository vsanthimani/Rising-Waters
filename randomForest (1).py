import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def randomForest(X_train, X_test, y_train, y_test, n_estimators=100, random_state=42):

    print("\n========== RANDOM FOREST MODEL BUILDING ==========")

    # Initialize Random Forest Classifier
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )

    print("[INFO] RandomForestClassifier initialized.")

    # Train the model
    model.fit(X_train, y_train)
    print("[INFO] Model training completed.")

    # Predict
    y_pred = model.predict(X_test)
    print("[INFO] Prediction completed.")

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    print("\nAccuracy :", accuracy)
    print("\nConfusion Matrix")
    print(cm)

    print("\nClassification Report")
    print(cr)

    return model, y_pred