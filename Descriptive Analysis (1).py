# ==========================================
# Descriptive Analysis using Pandas
# ==========================================

import pandas as pd

# Create a sample dataset
data = {
    "Name": ["Rahul", "Priya", "Amit", "Sneha", "Ravi"],
    "Age": [21, 22, 20, 23, 21],
    "Marks": [85, 90, 78, 88, 92],
    "City": ["Delhi", "Mumbai", "Hyderabad", "Chennai", "Bangalore"]
}

# Convert dictionary into DataFrame
df = pd.DataFrame(data)

# ------------------------------------------
# 1. Display the First Five Rows
# ------------------------------------------
print("=" * 50)
print("FIRST FIVE RECORDS")
print("=" * 50)
print(df.head())

# ------------------------------------------
# 2. Display Dataset Information
# ------------------------------------------
print("\n" + "=" * 50)
print("DATASET INFORMATION")
print("=" * 50)
df.info()

# ------------------------------------------
# 3. Statistical Summary
# ------------------------------------------
print("\n" + "=" * 50)
print("STATISTICAL SUMMARY")
print("=" * 50)
print(df.describe())

# ------------------------------------------
# 4. Data Types
# ------------------------------------------
print("\n" + "=" * 50)
print("DATA TYPES")
print("=" * 50)
print(df.dtypes)

# ------------------------------------------
# 5. Missing Values
# ------------------------------------------
print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)
print(df.isnull().sum())

# ------------------------------------------
# 6. Dataset Shape
# ------------------------------------------
print("\n" + "=" * 50)
print("DATASET SHAPE")
print("=" * 50)
print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])

# ------------------------------------------
# 7. Column Names
# ------------------------------------------
print("\n" + "=" * 50)
print("COLUMN NAMES")
print("=" * 50)
print(df.columns.tolist())

# ==========================================
# End of Descriptive Analysis
# ==========================================