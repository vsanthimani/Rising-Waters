from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

# Fit and transform training data
X_train = sc.fit_transform(X_train)

# Transform test data using the same scaler
X_test = sc.transform(X_test)