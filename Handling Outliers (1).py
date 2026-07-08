# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

# Create Sample Dataset
dataset = pd.DataFrame({
    'Age': [22, 25, 27, 29, 31, 35, 37, 40, 120, 18],
    'Salary': [25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 500000, 20000],
    'Experience': [1, 2, 3, 4, 5, 6, 7, 8, 20, 1],
    'Purchased': [0, 0, 0, 1, 1, 1, 1, 1, 1, 0]
})

# Independent Features
X = dataset.iloc[:, :-1].values

# Dependent Feature
y = dataset.iloc[:, -1].values

print("Independent Features (X)")
print(X)

print("\nDependent Feature (y)")
print(y)

# Display Original Dataset
print("\nOriginal Dataset")
print(dataset)

# Box Plot Before Handling Outliers
plt.boxplot(dataset["Salary"])
plt.title("Before Outlier Handling")
plt.show()

# Calculate Q1, Q3 and IQR
Q1 = dataset["Salary"].quantile(0.25)
Q3 = dataset["Salary"].quantile(0.75)

IQR = Q3 - Q1

# Calculate Lower and Upper Limits
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

print("\nLower Limit :", lower_limit)
print("Upper Limit :", upper_limit)

# Detect Outliers
outliers = dataset[
    (dataset["Salary"] < lower_limit) |
    (dataset["Salary"] > upper_limit)
]

print("\nDetected Outliers")
print(outliers)

# Apply Capping
dataset["Salary"] = dataset["Salary"].clip(lower_limit, upper_limit)

# Display Updated Dataset
print("\nDataset After Outlier Capping")
print(dataset)

# Box Plot After Handling Outliers
plt.boxplot(dataset["Salary"])
plt.title("After Outlier Handling")
plt.show()