import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Set up paths
data_paths = ['/mnt/data/', '/mnt/imported/data/']

def find_diabetes_dataset():
    """Search for diabetes dataset in standard Domino data paths."""
    possible_filenames = ['diabetes.csv', 'diabetes_data.csv', 'diabetes_dataset.csv']
    
    # Print all available files to help locate the dataset
    print("Listing available files in data directories:")
    for path in data_paths:
        if os.path.exists(path):
            print(f"\nFiles in {path}:")
            files = os.listdir(path)
            for file in files:
                print(f"  - {file}")
    
    # Try to find the dataset
    for path in data_paths:
        if os.path.exists(path):
            for filename in possible_filenames:
                file_path = os.path.join(path, filename)
                if os.path.exists(file_path):
                    print(f"\nFound dataset at {file_path}")
                    return file_path
                
            # If exact match not found, look for files containing 'diabetes'
            for file in os.listdir(path):
                if 'diabetes' in file.lower() and file.endswith(('.csv', '.xlsx', '.xls')):
                    file_path = os.path.join(path, file)
                    print(f"\nFound diabetes-related dataset at {file_path}")
                    return file_path
    
    return None

def analyze_dataset(file_path):
    """Analyze the diabetes dataset and evaluate its usefulness for modeling."""
    print(f"Analyzing dataset: {file_path}")
    
    # Determine file type and read accordingly
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    # Basic dataset information
    print("\n=== DATASET OVERVIEW ===")
    print(f"Dataset shape: {df.shape}")
    print(f"Number of samples: {df.shape[0]}")
    print(f"Number of features: {df.shape[1]}")
    
    # Print column names
    print("\n=== COLUMNS ===")
    for col in df.columns:
        print(f"  - {col}")
    
    # Check for target variable
    target_candidates = [col for col in df.columns if any(t in col.lower() for t in 
                                                          ['diabet', 'outcome', 'target', 'class', 'label'])]
    if target_candidates:
        print(f"\nPossible target columns: {target_candidates}")
    
    # Data summary
    print("\n=== DATA SUMMARY ===")
    print(df.describe().transpose())
    
    # Check for missing values
    missing_values = df.isnull().sum()
    print("\n=== MISSING VALUES ===")
    if missing_values.sum() == 0:
        print("No missing values found.")
    else:
        print(missing_values[missing_values > 0])
    
    # Check data types
    print("\n=== DATA TYPES ===")
    print(df.dtypes)
    
    # Analyze class distribution if target is identified
    if target_candidates:
        target_col = target_candidates[0]  # Take first candidate as default
        print(f"\n=== TARGET DISTRIBUTION ({target_col}) ===")
        target_dist = df[target_col].value_counts(normalize=True) * 100
        print(target_dist)
        
        # Check for class imbalance
        if len(target_dist) > 1:
            min_class_pct = target_dist.min()
            print(f"Minimum class percentage: {min_class_pct:.2f}%")
            if min_class_pct < 10:
                print("WARNING: Severe class imbalance detected")
            elif min_class_pct < 30:
                print("CAUTION: Moderate class imbalance detected")
    
    # Save summary plots
    print("\n=== CREATING VISUALIZATIONS ===")
    
    # Select numeric columns for correlation analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 1:
        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png')
        print("Saved correlation heatmap as correlation_heatmap.png")
        
        # Distribution of numeric features
        plt.figure(figsize=(15, 10))
        df[numeric_cols].hist(bins=20, figsize=(15, 10))
        plt.tight_layout()
        plt.savefig('feature_distributions.png')
        print("Saved feature distributions as feature_distributions.png")
    
    return df

def transform_data(df):
    """Perform necessary data transformations for model training."""
    print("\n=== DATA TRANSFORMATION ===")
    
    # Create a copy to avoid modifying original data
    df_transformed = df.copy()
    
    # Identify target column if possible
    target_candidates = [col for col in df.columns if any(t in col.lower() for t in 
                                                          ['diabet', 'outcome', 'target', 'class', 'label'])]
    target_col = target_candidates[0] if target_candidates else None
    
    # Handle missing values if any
    if df_transformed.isnull().sum().sum() > 0:
        print("Handling missing values...")
        # For numeric columns, fill with median
        numeric_cols = df_transformed.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_transformed[col].isnull().sum() > 0:
                df_transformed[col] = df_transformed[col].fillna(df_transformed[col].median())
        
        # For categorical columns, fill with mode
        cat_cols = df_transformed.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if df_transformed[col].isnull().sum() > 0:
                df_transformed[col] = df_transformed[col].fillna(df_transformed[col].mode()[0])
    
    # Handle potential outliers in numeric features (using IQR method)
    print("Checking for outliers...")
    numeric_cols = df_transformed.select_dtypes(include=[np.number]).columns
    if target_col and target_col in numeric_cols:
        numeric_cols = [col for col in numeric_cols if col != target_col]
    
    outlier_summary = {}
    for col in numeric_cols:
        Q1 = df_transformed[col].quantile(0.25)
        Q3 = df_transformed[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((df_transformed[col] < lower_bound) | (df_transformed[col] > upper_bound)).sum()
        if outliers > 0:
            outlier_pct = (outliers / len(df_transformed)) * 100
            outlier_summary[col] = f"{outliers} ({outlier_pct:.2f}%)"
            
            # Cap outliers
            df_transformed[col] = np.where(df_transformed[col] < lower_bound, lower_bound, df_transformed[col])
            df_transformed[col] = np.where(df_transformed[col] > upper_bound, upper_bound, df_transformed[col])
    
    if outlier_summary:
        print("Outliers detected and capped:")
        for col, summary in outlier_summary.items():
            print(f"  - {col}: {summary}")
    else:
        print("No significant outliers detected")
    
    # Standardize numeric features (excluding target)
    print("Standardizing numeric features...")
    if numeric_cols:
        scaler = StandardScaler()
        df_transformed[numeric_cols] = scaler.fit_transform(df_transformed[numeric_cols])
    
    # Handle categorical features if any
    cat_cols = df_transformed.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        print(f"Encoding {len(cat_cols)} categorical features...")
        df_transformed = pd.get_dummies(df_transformed, columns=cat_cols, drop_first=True)
    
    # Prepare train/test split if target identified
    if target_col:
        print(f"Creating train/test split using target: {target_col}")
        X = df_transformed.drop(columns=[target_col])
        y = df_transformed[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        print(f"Training set size: {len(X_train)}")
        print(f"Testing set size: {len(X_test)}")
        
        # Save processed datasets
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        train_data.to_csv('diabetes_train.csv', index=False)
        test_data.to_csv('diabetes_test.csv', index=False)
        print("Saved processed train/test datasets")
    else:
        print("No target column identified. Saving single transformed dataset.")
        df_transformed.to_csv('diabetes_transformed.csv', index=False)
    
    # Summary of transformation
    print("\n=== TRANSFORMATION SUMMARY ===")
    print(f"Original dataset shape: {df.shape}")
    print(f"Transformed dataset shape: {df_transformed.shape}")
    
    return df_transformed

def main():
    """Main function to run the analysis and transformation."""
    print("=== DIABETES DATASET ANALYSIS ===")
    
    # Find the dataset
    file_path = find_diabetes_dataset()
    
    if file_path is None:
        print("\nERROR: Could not find a diabetes dataset in the standard data paths.")
        print("Please ensure the dataset is available in one of these paths:")
        for path in data_paths:
            print(f"  - {path}")
        return
    
    # Analyze the dataset
    df = analyze_dataset(file_path)
    
    # Transform the data
    df_transformed = transform_data(df)
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("The dataset has been analyzed and transformed.")
    print("Check the generated CSV files and images for the results.")
    
    # Evaluation of dataset usefulness
    print("\n=== DATASET USEFULNESS EVALUATION ===")
    
    # Check size
    if df.shape[0] < 100:
        print("CONCERN: The dataset is very small, which may limit model performance.")
    elif df.shape[0] < 500:
        print("NOTE: The dataset is relatively small. Consider techniques to address potential overfitting.")
    else:
        print("POSITIVE: Dataset size appears adequate for initial modeling.")
    
    # Check features
    if df.shape[1] < 5:
        print("CONCERN: The dataset has few features, which may limit predictive power.")
    else:
        print(f"POSITIVE: The dataset has {df.shape[1]} features, which should provide good signal for modeling.")
    
    # Final recommendation
    print("\n=== RECOMMENDATION ===")
    print("The transformed dataset is ready for model training. Consider:")
    print("1. Starting with simpler models (logistic regression, random forest) before complex ones")
    print("2. Using cross-validation to ensure model stability")
    print("3. Monitoring for overfitting, especially if the dataset is small")
    print("4. Feature importance analysis to identify the most predictive variables")

if __name__ == "__main__":
    main() 