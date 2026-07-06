import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Model Library Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor

# Set styling for visualizations
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ==========================================
# EPIC 1 & 2: DATA COLLECTION & ANALYSIS
# ==========================================
def load_and_analyze_data(filepath):
    print("--- Epic 1 & 2: Loading and Exploring Data ---")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Please place your dataset at {filepath}")
        
    # Read the excel workbook file directly (Requires openpyxl)
    df = pd.read_excel(filepath)
    print(f"Dataset Shape: {df.shape}")
    print("\nDataset Info:")
    print(df.info())
    
    print("\nDescriptive Statistical Summary:")
    print(df.describe().T)
    
    # Identify target column dynamically
    target_col = 'FloodProbability' if 'FloodProbability' in df.columns else df.columns[-1]
    feature_cols = [col for col in df.columns if col != target_col and col != 'id']
    
    # Univariate Analysis (Distribution Plot)
    plt.figure()
    sns.histplot(df[target_col], kde=True, bins=30)
    plt.title(f'Univariate Analysis: Distribution of {target_col}')
    plt.savefig('univariate_target_distribution.png')
    plt.close()
    
    # Multivariate Analysis (Correlation Heatmap)
    plt.figure(figsize=(12, 10))
    # Filter numerical columns for correlation calculation
    num_cols = df[feature_cols + [target_col]].select_dtypes(include=[np.number]).columns.tolist()
    correlation_matrix = df[num_cols].corr()
    
    # Isolate top 10 features correlated with target
    top_features = correlation_matrix[target_col].abs().sort_values(ascending=False).index[:11]
    
    # Pull directly from our computed correlation slice to completely avoid mixed-type column bugs
    sns.heatmap(correlation_matrix.loc[top_features, top_features], annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Multivariate Analysis: Feature Correlation Heatmap')
    plt.savefig('multivariate_correlation_matrix.png')
    plt.close()
    
    print("\n[✓] Exploratory data analysis graphs saved to local directory.")
    return df, feature_cols, target_col

# ==========================================
# EPIC 3: DATA PRE-PROCESSING
# ==========================================
def preprocess_data(df, feature_cols, target_col):
    print("\n--- Epic 3: Data Pre-Processing ---")
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # 1. Handle Missing Values using column medians
    if X.isnull().sum().sum() > 0:
        print("Handling missing values using median imputation...")
        X = X.fillna(X.median())
    else:
        print("No missing values detected.")
        
    # 2. Convert Categorical to Numerical (One-Hot Encoding if string features are present)
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if len(categorical_cols) > 0:
        print(f"Encoding categorical features: {categorical_cols}")
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    else:
        print("No structural categorical features to transform.")
        
    # 3. Split Dataset First (Prevents Data Leakage into evaluation benchmarks)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Initial splits - Train: {X_train.shape}, Test: {X_test.shape}")
    
    # 4. Detect and Treat Outliers *strictly* inside the Training Data split
    print("Detecting and filtering anomalies from training data...")
    iso = IsolationForest(contamination=0.01, random_state=42)
    train_outliers = iso.fit_predict(X_train)
    clean_train_mask = train_outliers != -1
    
    X_train = X_train[clean_train_mask]
    y_train = y_train[clean_train_mask]
    print(f"Final training size after outlier elimination: {X_train.shape}")
    
    # 5. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save schema state for structural alignment during runtime parsing
    feature_structure = X.columns.tolist()
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_structure

# ==========================================
# EPIC 4: MODEL BUILDING & COMPARISON
# ==========================================
def build_and_evaluate_models(X_train, X_test, y_train, y_test):
    print("\n--- Epic 4: Model Building & Evaluation ---")
    
    models = {
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=7, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
    }
    
    performance_metrics = {}
    
    for name, model in models.items():
        print(f"Training {name} Model...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        performance_metrics[name] = {"Model": model, "RMSE": rmse, "MAE": mae, "R2 Score": r2}
        print(f"{name} -> R2 Score: {r2:.4f} | RMSE: {rmse:.4f}")

    # Compare Performances
    comparison_df = pd.DataFrame(performance_metrics).T.drop(columns=["Model"])
    print("\nModel Comparison Matrix Summary:")
    print(comparison_df.to_string())
    
    # Select the optimal configurations
    best_model_name = comparison_df["R2 Score"].idxmax()
    best_model_meta = performance_metrics[best_model_name]
    print(f"\n[★] Best Performing Model Chosen: {best_model_name} with R2 Score: {best_model_meta['R2 Score']:.4f}")
    
    return best_model_meta["Model"], best_model_name

# ==========================================
# EXECUTION PIPELINE CONTROL
# ==========================================
if __name__ == "__main__":
    # Point this to your downloaded Excel workbook file
    DATASET_PATH = "flood dataset.xlsx" 
    
    try:
        # Execute Pipeline Steps
        df, features, target = load_and_analyze_data(DATASET_PATH)
        X_train, X_test, y_train, y_test, scaler, feat_cols_list = preprocess_data(df, features, target)
        champion_model, model_name = build_and_evaluate_models(X_train, X_test, y_train, y_test)
        
        # Save Artifact Bundles for operational deployments
        print("\nSaving selected architecture artifacts to files...")
        with open("flood_model.pkl", "wb") as m_file:
            pickle.dump(champion_model, m_file)
        with open("scaler.pkl", "wb") as s_file:
            pickle.dump(scaler, s_file)
        with open("features.pkl", "wb") as f_file:
            pickle.dump(feat_cols_list, f_file)
            
        print("[✓] Process complete. Artifact files ('flood_model.pkl', 'scaler.pkl', 'features.pkl') generated successfully.")
        
    except FileNotFoundError as e:
        print(f"\n[!] Execution halted: {e}")
        print("Generating a fallback mock workbook 'flood dataset.xlsx' for direct structural testing...")
        
        # Create a mock spreadsheet dataset to keep execution continuous if file is missing
        mock_data = pd.DataFrame(np.random.randint(20, 150, size=(1000, 6)), 
                                 columns=['Annual_Rainfall', 'Cloud_Visibility', 'Jun_Sep_Rainfall', 'River_Management', 'Deforestation', 'FloodProbability'])
        mock_data['FloodProbability'] = (mock_data['Annual_Rainfall'] * 0.4 + mock_data['Jun_Sep_Rainfall'] * 0.6) / 200.0
        mock_data.to_excel("flood dataset.xlsx", index=False)
        print("[!] Synthetic file 'flood dataset.xlsx' created. Please run the script again.")