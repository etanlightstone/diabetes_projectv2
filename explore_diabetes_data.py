#!/usr/bin/env python
# Data exploration script for the diabetes dataset

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configure visualization settings
plt.style.use('seaborn-whitegrid')
sns.set(font_scale=1.2)
pd.set_option('display.max_columns', None)

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
        # Let's check if any CSV files exist
        for data_dir in data_dirs:
            if os.path.exists(data_dir):
                csv_files = []
                for root, _, files in os.walk(data_dir):
                    for file in files:
                        if file.endswith('.csv'):
                            csv_files.append(os.path.join(root, file))
                if csv_files:
                    print(f"Found CSV files: {csv_files}")
    
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

def explore_data(df):
    """Explore the dataset and generate summary statistics and visualizations"""
    print("\n=== DATASET OVERVIEW ===")
    print(f"Dataset shape: {df.shape}")
    
    print("\n=== COLUMN NAMES ===")
    print(df.columns.tolist())
    
    print("\n=== FIRST 5 ROWS ===")
    print(df.head())
    
    print("\n=== DATA TYPES ===")
    print(df.dtypes)
    
    print("\n=== MISSING VALUES ===")
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_info = pd.DataFrame({
        'Count': missing,
        'Percent': missing_percent
    })
    print(missing_info[missing_info['Count'] > 0])
    
    print("\n=== SUMMARY STATISTICS ===")
    print(df.describe().T)
    
    # Create output directory for plots
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    
    # Identify potential target column for diabetes prediction
    potential_target = None
    for col in df.columns:
        if 'diabet' in col.lower() or 'outcome' in col.lower() or 'target' in col.lower():
            potential_target = col
            print(f"\nPotential target column identified: '{col}'")
            print(f"Target distribution:\n{df[col].value_counts(normalize=True)*100}")
            
            # Plot target distribution
            plt.figure(figsize=(8, 5))
            sns.countplot(x=col, data=df)
            plt.title(f'Distribution of {col}')
            plt.savefig(output_dir / f"{col}_distribution.png")
            break
    
    # Plot histograms for numeric features
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for i in range(0, len(numeric_cols), 3):
        cols_to_plot = numeric_cols[i:i+3]
        if len(cols_to_plot) > 0:
            plt.figure(figsize=(15, 5))
            for j, col in enumerate(cols_to_plot):
                plt.subplot(1, len(cols_to_plot), j+1)
                sns.histplot(df[col], kde=True)
                plt.title(f'Distribution of {col}')
            plt.tight_layout()
            plt.savefig(output_dir / f"numeric_features_{i}.png")
    
    # Check for correlations
    plt.figure(figsize=(12, 10))
    correlation = df.select_dtypes(include=['int64', 'float64']).corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f")
    plt.title('Feature Correlation Matrix')
    plt.savefig(output_dir / "correlation_matrix.png")
    
    # If we have a target variable, analyze feature importance
    if potential_target is not None and potential_target in df.columns:
        print("\n=== FEATURE ANALYSIS BY TARGET ===")
        for col in numeric_cols:
            if col != potential_target:
                print(f"\nFeature: {col}")
                try:
                    print(df.groupby(potential_target)[col].mean())
                except:
                    print(f"Could not analyze {col} by target")
    
    return output_dir

def evaluate_dataset_quality(df, target_col=None):
    """Evaluate the dataset quality for machine learning"""
    issues = []
    recommendations = []
    
    # Check dataset size
    if len(df) < 100:
        issues.append("Dataset is very small for machine learning")
        recommendations.append("Consider collecting more data or using techniques for small datasets")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        issues.append("Dataset contains missing values")
        recommendations.append("Consider imputation strategies for missing values")
    
    # Check for categorical columns that might need encoding
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        issues.append(f"Categorical features identified: {list(categorical_cols)}")
        recommendations.append("Consider one-hot encoding or label encoding for categorical features")
    
    # Check for imbalanced target (if identified)
    if target_col is not None and target_col in df.columns:
        target_counts = df[target_col].value_counts(normalize=True)
        if target_counts.min() < 0.2:
            issues.append(f"Imbalanced target variable: {dict(target_counts)}")
            recommendations.append("Consider using class weights, SMOTE, or other balancing techniques")
    
    # Check for high correlations between features
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    corr_matrix = df[numeric_cols].corr().abs()
    high_corr_features = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.8:
                high_corr_features.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    
    if high_corr_features:
        issues.append(f"High correlation between features: {high_corr_features}")
        recommendations.append("Consider removing one of each highly correlated feature pair")
    
    # Check for features with high skewness
    for col in numeric_cols:
        skewness = df[col].skew()
        if abs(skewness) > 1:
            issues.append(f"Feature '{col}' has high skewness: {skewness}")
            recommendations.append(f"Consider applying transformation (log, sqrt, etc.) to '{col}'")
    
    return issues, recommendations

def main():
    # Find and explore the diabetes dataset
    data_files = find_dataset()
    
    if not data_files:
        print("No diabetes dataset found. Please ensure the dataset is available in one of the specified directories.")
        return
    
    # Use the first matching file
    dataset_path = data_files[0]
    print(f"Using dataset: {dataset_path}")
    
    try:
        # Load and explore the data
        df = load_dataset(dataset_path)
        output_dir = explore_data(df)
        
        # Try to identify the target column
        potential_target = None
        for col in df.columns:
            if 'diabet' in col.lower() or 'outcome' in col.lower() or 'target' in col.lower():
                potential_target = col
                break
        
        # Evaluate dataset quality
        issues, recommendations = evaluate_dataset_quality(df, potential_target)
        
        print("\n=== DATASET QUALITY ASSESSMENT ===")
        if issues:
            print("Identified issues:")
            for i, issue in enumerate(issues, 1):
                print(f"{i}. {issue}")
        else:
            print("No significant issues identified.")
        
        print("\nRecommendations:")
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}")
        else:
            print("No specific recommendations needed.")
        
        print(f"\nExploration results saved to {output_dir}")
        
    except Exception as e:
        print(f"Error analyzing dataset: {str(e)}")

if __name__ == "__main__":
    main() 