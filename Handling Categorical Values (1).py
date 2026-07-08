# ==========================================
# Handling Categorical Values
# Feature Mapping and Label Encoding
# ==========================================

# Import Libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ------------------------------------------
# Create Sample Dataset
# ------------------------------------------

dataset = pd.DataFrame({
    "Education": ["Low", "Medium", "High", "Medium", "Low", "High"],
    "Gender": ["Male", "Female", "Female", "Male", "Male", "Female"],
    "Occupation": ["Student", "Working", "Pensioner", "Working", "Student", "Pensioner"],
    "Purchased": ["Yes", "No", "Yes", "Yes", "No", "Yes"]
})

# ------------------------------------------
# Display Original Dataset
# ------------------------------------------

print("=" * 60)
print("ORIGINAL DATASET")
print("=" * 60)
print(dataset)

# ------------------------------------------
# Feature Mapping
# ------------------------------------------

education_mapping = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

dataset["Education"] = dataset["Education"].map(education_mapping)

print("\n" + "=" * 60)
print("AFTER FEATURE MAPPING")
print("=" * 60)
print(dataset)

# ------------------------------------------
# Label Encoding
# ------------------------------------------

label_encoder = LabelEncoder()

dataset["Gender"] = label_encoder.fit_transform(dataset["Gender"])
dataset["Occupation"] = label_encoder.fit_transform(dataset["Occupation"])
dataset["Purchased"] = label_encoder.fit_transform(dataset["Purchased"])

print("\n" + "=" * 60)
print("AFTER LABEL ENCODING")
print("=" * 60)
print(dataset)

# ------------------------------------------
# Independent and Dependent Variables
# ------------------------------------------

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print("\n" + "=" * 60)
print("INDEPENDENT FEATURES (X)")
print("=" * 60)
print(X)

print("\n" + "=" * 60)
print("DEPENDENT FEATURE (y)")
print("=" * 60)
print(y)

# ------------------------------------------
# Data Types
# ------------------------------------------

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)
print(dataset.dtypes)

# ==========================================
# End of Program
# ==========================================