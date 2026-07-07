import joblib

# Save the trained model
joblib.dump(p4, 'floods.save')

# Save the StandardScaler
joblib.dump(sc, 'transform.save')