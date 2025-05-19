#!/usr/bin/env python
# Data transformation script for the diabetes dataset

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

def find_dataset(data_dirs=['/mnt/data', '/mnt/imported/data']):
    """Search for diabetes dataset in the provided directories"""
    print("Searching for diabetes dataset...")
    potential_files = []
    
    for data_dir in data_dirs:
        if not os.path.exists(data_dir):
            print(f"Directory {data_dir} does not exist")
            continue
            
        for root, _, files in os.walk(data_dir):
            for file in files:
                if 'diabetes' in file.lower() and file.endswith(('.csv', '.xlsx', '.xls')):
                    file_path = os.path.join(root, file)
                    potential_files.append(file_path)
    
    if not potential_files:
        print("No diabetes dataset found in the specified directories.")
        
    return potential_files

def load_dataset(file_path):
    """Load the dataset from the given path"""
    print(f"Loading dataset from: {file_path}")
    
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    return df

def identify_target_column(df):
    """Try to identify the target column for diabetes prediction"""
    potential_target = None
    for col in df.columns:
        if 'diabet' in col.lower() or 'outcome' in col.lower() or 'target' in col.lower():
            potential_target = col
            print(f"Identified target column: '{col}'")
            print(f"Target distribution: {df[col].value_counts()}")
            break
    
    if potential_target is None:
        print("Could not automatically identify a target column.")
        
    return potential_target

def preprocess_data(df, target_col=None):
    """Apply preprocessing transformations to prepare data for modeling"""
    print("\n=== PREPROCESSING DATA ===")
    transformations_applied = []
    
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Step 1: Handle missing values
    if processed_df.isnull().sum().sum() > 0:
        print("Handling missing values...")
        numeric_cols = processed_df.select_dtypes(include=['int64', 'float64']).columns
        
        # For numeric columns, impute with median
        if len(numeric_cols) > 0:
            numeric_imputer = SimpleImputer(strategy='median')
            processed_df[numeric_cols] = numeric_imputer.fit_transform(processed_df[numeric_cols])
            transformations_applied.append("Imputed missing numeric values with median")
        
        # For categorical columns, impute with most frequent value
        cat_cols = processed_df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            for col in cat_cols:
                processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0])
            transformations_applied.append("Imputed missing categorical values with mode")
    
    # Step 2: Handle zeros in features where zeros are unlikely valid values
    # Common in diabetes datasets: features like Glucose, BloodPressure, SkinThickness, 
    # Insulin, BMI should not be zero
    zero_invalid_columns = []
    for col in processed_df.columns:
        # Look for potential health measurement columns that shouldn't be zero
        col_lower = col.lower()
        if any(term in col_lower for term in ['glucose', 'blood', 'pressure', 'skin', 'thickness', 
                                             'insulin', 'bmi', 'age']):
            if (processed_df[col] == 0).sum() > 0:
                zero_invalid_columns.append(col)
    
    if zero_invalid_columns:
        print(f"Handling zeros in columns where zeros are likely invalid: {zero_invalid_columns}")
        for col in zero_invalid_columns:
            # Replace zeros with NaN and then impute with median of non-zero values
            col_median = processed_df.loc[processed_df[col] != 0, col].median()
            processed_df.loc[processed_df[col] == 0, col] = col_median
        transformations_applied.append(f"Replaced zeros with median in {len(zero_invalid_columns)} columns")
    
    # Step 3: Handle outliers
    numeric_cols = processed_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if target_col and target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    if numeric_cols:
        print("Handling outliers in numeric columns...")
        for col in numeric_cols:
            # Calculate IQR
            Q1 = processed_df[col].quantile(0.25)
            Q3 = processed_df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers
            outliers = ((processed_df[col] < lower_bound) | (processed_df[col] > upper_bound)).sum()
            
            if outliers > 0:
                # Cap outliers
                processed_df[col] = processed_df[col].clip(lower_bound, upper_bound)
                print(f"  - Capped {outliers} outliers in '{col}'")
        
        transformations_applied.append("Capped outliers using IQR method")
    
    # Step 4: Apply transformations to skewed features
    for col in numeric_cols:
        skewness = processed_df[col].skew()
        if abs(skewness) > 1:
            print(f"Applying transformation to skewed feature '{col}' (skewness: {skewness:.2f})")
            
            # Apply log transformation to positive skewed data
            if skewness > 0 and processed_df[col].min() > 0:
                processed_df[f"{col}_orig"] = processed_df[col]  # keep original
                processed_df[col] = np.log1p(processed_df[col])
                transformations_applied.append(f"Applied log transformation to '{col}'")
            else:
                # Use Yeo-Johnson transformation which works with negative values
                pt = PowerTransformer(method='yeo-johnson')
                processed_df[f"{col}_orig"] = processed_df[col]  # keep original
                processed_df[col] = pt.fit_transform(processed_df[[col]])
                transformations_applied.append(f"Applied Yeo-Johnson transformation to '{col}'")
    
    # Step 5: Feature scaling
    print("Applying feature scaling...")
    feature_cols = [col for col in numeric_cols if not col.endswith('_orig')]
    if feature_cols:
        scaler = StandardScaler()
        processed_df[feature_cols] = scaler.fit_transform(processed_df[feature_cols])
        transformations_applied.append("Applied StandardScaler to numeric features")
    
    return processed_df, transformations_applied

def handle_class_imbalance(df, target_col):
    """Handle class imbalance in the target variable"""
    if target_col and target_col in df.columns:
        value_counts = df[target_col].value_counts()
        min_class_pct = value_counts.min() / len(df) * 100
        
        print(f"\n=== CLASS BALANCE ===")
        print(f"Target distribution: {value_counts}")
        print(f"Minimum class percentage: {min_class_pct:.2f}%")
        
        # If the dataset is imbalanced (less than 30% in the minority class)
        if min_class_pct < 30:
            print("Applying SMOTE to balance the dataset...")
            
            # Separate features and target
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            # Apply SMOTE
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            # Combine back into a dataframe
            balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
            balanced_df[target_col] = y_resampled
            
            print(f"After SMOTE: {balanced_df[target_col].value_counts()}")
            return balanced_df, True
        else:
            print("Class distribution is relatively balanced. No resampling needed.")
            return df, False
    
    return df, False

def main():
    # Find diabetes dataset
    data_files = find_dataset()
    
    if not data_files:
        print("No diabetes dataset found. Please ensure the dataset is available in one of the specified directories.")
        return
    
    # Use the first matching file
    dataset_path = data_files[0]
    print(f"Using dataset: {dataset_path}")
    
    try:
        # Load the data
        df = load_dataset(dataset_path)
        print(f"Original dataset shape: {df.shape}")
        
        # Identify target column
        target_col = identify_target_column(df)
        
        # Apply preprocessing transformations
        processed_df, transformations = preprocess_data(df, target_col)
        
        # Handle class imbalance if we have a target column
        if target_col:
            balanced_df, was_balanced = handle_class_imbalance(processed_df, target_col)
            if was_balanced:
                processed_df = balanced_df
                transformations.append("Applied SMOTE to balance classes")
        
        # Save transformed dataset
        output_dir = Path("./processed_data")
        output_dir.mkdir(exist_ok=True)
        
        transformed_path = output_dir / "diabetes_transformed.csv"
        processed_df.to_csv(transformed_path, index=False)
        
        print(f"\n=== TRANSFORMATION SUMMARY ===")
        for i, transform in enumerate(transformations, 1):
            print(f"{i}. {transform}")
        
        print(f"\nTransformed dataset shape: {processed_df.shape}")
        print(f"Saved transformed dataset to: {transformed_path}")
        
    except Exception as e:
        print(f"Error processing dataset: {str(e)}")

if __name__ == "__main__":
    main() 