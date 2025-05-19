import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Set up visualization
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")
pd.set_option('display.max_columns', None)

# Path to the dataset
dataset_path = '/mnt/imported/data/diabetes_datafiles/diabetes_dataset.csv'
output_dir = '/mnt/data/diabetes_project'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the data
print(f"Loading dataset from {dataset_path}")
df = pd.read_csv(dataset_path)

print(f"Original dataset shape: {df.shape}")
print("\nFeature distributions:")
for col in df.columns:
    if col != 'is_diabetic':
        print(f"{col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}, std={df[col].std():.2f}")

# Data Transformation Steps:
print("\n### Data Transformation Steps ###")

# 1. Check for outliers and cap them if necessary
print("\n1. Checking for outliers...")
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_features.remove('is_diabetic')  # Remove target variable

for col in numeric_features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_count = sum((df[col] < lower_bound) | (df[col] > upper_bound))
    outlier_pct = outliers_count / len(df) * 100
    print(f"{col}: {outliers_count} outliers ({outlier_pct:.2f}%)")
    
    # Cap outliers
    if outlier_pct > 1:  # Only cap if significant number of outliers
        df[col] = df[col].clip(lower_bound, upper_bound)
        print(f"   - Capped {col} between {lower_bound:.2f} and {upper_bound:.2f}")

# 2. Create derived features
print("\n2. Creating derived features...")

# BMI approximation (weight is already in dataset, but we don't have height)
# This is a rough proxy based on weight since we can't calculate true BMI
df['weight_category'] = pd.qcut(df['weight'], 4, labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

# Exercise level - combine hours and intensity
df['exercise_level'] = df['hrs_exercise_wk'] * df['exercise_intensity']
print(f"Created exercise_level feature: min={df['exercise_level'].min():.2f}, max={df['exercise_level'].max():.2f}")

# Income category 
df['income_category'] = pd.qcut(df['annual_income'], 4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])

# 3. One-hot encode categorical features
print("\n3. One-hot encoding categorical features...")
categorical_cols = ['weight_category', 'income_category']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print(f"Shape after one-hot encoding: {df_encoded.shape}")

# 4. Feature scaling
print("\n4. Feature scaling...")
# Select features for scaling (exclude target variable and categorical variables)
features_to_scale = ['calories_wk', 'hrs_exercise_wk', 'exercise_intensity', 
                     'annual_income', 'weight', 'exercise_level']

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the features
df_encoded[features_to_scale] = scaler.fit_transform(df_encoded[features_to_scale])

# Save the scaler for later use
joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
print(f"Scaler saved to {output_dir}/scaler.pkl")

# 5. Split the data into training and testing sets
print("\n5. Splitting data into train and test sets...")
X = df_encoded.drop('is_diabetic', axis=1)
y = df_encoded['is_diabetic']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")
print(f"Class distribution in training set: {y_train.value_counts().to_dict()}")
print(f"Class distribution in testing set: {y_test.value_counts().to_dict()}")

# 6. Save the processed data
print("\n6. Saving processed data...")
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

train_data.to_csv(os.path.join(output_dir, 'diabetes_train.csv'), index=False)
test_data.to_csv(os.path.join(output_dir, 'diabetes_test.csv'), index=False)

# Save feature names for later use
feature_names = X.columns.tolist()
with open(os.path.join(output_dir, 'feature_names.txt'), 'w') as f:
    f.write('\n'.join(feature_names))

print(f"Training data saved to {output_dir}/diabetes_train.csv")
print(f"Testing data saved to {output_dir}/diabetes_test.csv")
print(f"Feature names saved to {output_dir}/feature_names.txt")

# 7. Feature importance analysis
print("\n7. Feature correlation with target...")
# Calculate correlations using original numeric features only
numeric_df = df.select_dtypes(include=['int64', 'float64'])
correlations = numeric_df.corr()['is_diabetic'].sort_values(ascending=False)
print(correlations)

# 8. Summary and recommendations
print("\n### Dataset Analysis Summary ###")
print(f"- Dataset size: {df.shape[0]} samples with {df.shape[1]} original features")
print(f"- Processed features: {X.shape[1]} features after transformation")
print(f"- Class imbalance: {df['is_diabetic'].mean()*100:.1f}% positive class (diabetic)")
print(f"- Strongest positive correlations with diabetes: {correlations[1:4].index.tolist()}")
print(f"- Strongest negative correlations with diabetes: {correlations[-3:].index.tolist()}") 