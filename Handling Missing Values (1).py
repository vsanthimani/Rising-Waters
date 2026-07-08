# ==========================================
# Handling Missing Values using Pandas
# ==========================================

# Import pandas
import pandas as pd

# ------------------------------------------
# Create a Sample Dataset
# ------------------------------------------

data = {
    "Name": ["Rahul", "Priya", "Amit", None, "Sneha"],
    "Age": [21, None, 20, 22, 23],
    "Marks": [85, 90, None, 88, 92],
    "City": ["Delhi", "Mumbai", None, "Chennai", "Bangalore"]
}

# Convert dictionary into DataFrame
df = pd.DataFrame(data)

# ------------------------------------------
# Display Dataset
# ------------------------------------------

print("=" * 50)
print("ORIGINAL DATASET")
print("=" * 50)
print(df)

# ------------------------------------------
# Check Missing Values
# ------------------------------------------

print("\n" + "=" * 50)
print("MISSING VALUES IN EACH COLUMN")
print("=" * 50)
print(df.isnull().sum())

# ------------------------------------------
# Check if Any Missing Values Exist
# ------------------------------------------

print("\n" + "=" * 50)
print("DOES THE DATASET CONTAIN MISSING VALUES?")
print("=" * 50)
print(df.isnull().any())

# ------------------------------------------
# Fill Missing Numeric Values with Mean
# ------------------------------------------

df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Marks"] = df["Marks"].fillna(df["Marks"].mean())

# ------------------------------------------
# Fill Missing Categorical Values with Mode
# ------------------------------------------

df["Name"] = df["Name"].fillna(df["Name"].mode()[0])
df["City"] = df["City"].fillna(df["City"].mode()[0])

# ------------------------------------------
# Display Updated Dataset
# ------------------------------------------

print("\n" + "=" * 50)
print("DATASET AFTER HANDLING MISSING VALUES")
print("=" * 50)
print(df)

# ------------------------------------------
# Verify Missing Values Again
# ------------------------------------------

print("\n" + "=" * 50)
print("MISSING VALUES AFTER CLEANING")
print("=" * 50)
print(df.isnull().sum())