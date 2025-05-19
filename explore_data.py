import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set up visualization
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")
pd.set_option('display.max_columns', None)

# First, let's find the diabetes dataset in the /mnt/data and /mnt/imported/data directories
print("### Exploring data directories ###")
for directory in ['/mnt/data', '/mnt/imported/data']:
    if os.path.exists(directory):
        print(f"\nContents of {directory}:")
        for root, dirs, files in os.walk(directory):
            level = root.replace(directory, '').count(os.sep)
            indent = ' ' * 4 * level
            print(f"{indent}{os.path.basename(root)}/")
            file_indent = ' ' * 4 * (level + 1)
            for f in files:
                print(f"{file_indent}{f}")
    else:
        print(f"\nDirectory {directory} does not exist")

# Based on standard dataset names, look for potential diabetes datasets
data_files = []
for directory in ['/mnt/data', '/mnt/imported/data']:
    if os.path.exists(directory):
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(('.csv', '.xlsx', '.parquet')):
                    if any(keyword in file.lower() for keyword in ['diab', 'glucose', 'pima']):
                        data_files.append(os.path.join(root, file))

# If we found potential diabetes datasets, analyze the first one
if data_files:
    print(f"\n### Found {len(data_files)} potential diabetes dataset(s) ###")
    for data_file in data_files:
        print(f"\nAnalyzing file: {data_file}")
        
        try:
            # Load the data
            if data_file.endswith('.csv'):
                df = pd.read_csv(data_file)
            elif data_file.endswith('.xlsx'):
                df = pd.read_excel(data_file)
            elif data_file.endswith('.parquet'):
                df = pd.read_parquet(data_file)
                
            # Basic dataset information
            print(f"\nDataset Shape: {df.shape}")
            print("\nFirst 5 rows:")
            print(df.head().to_string())
            
            print("\nData Types:")
            print(df.dtypes)
            
            print("\nSummary Statistics:")
            print(df.describe().to_string())
            
            print("\nMissing Values:")
            missing = df.isnull().sum()
            print(missing[missing > 0].to_string())
            
            # Check for class distribution if outcome column exists
            potential_target_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['outcome', 'target', 'class', 'diabetic'])]
            if potential_target_cols:
                target_col = potential_target_cols[0]
                print(f"\nClass Distribution for '{target_col}':")
                print(df[target_col].value_counts())
                print(f"Class proportion (1): {df[target_col].mean():.4f}")
            
            print("\nCorrelation Matrix:")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 1:
                corr = df[numeric_cols].corr()
                print(corr.to_string())
        
        except Exception as e:
            print(f"Error analyzing {data_file}: {str(e)}")
else:
    print("\nNo dataset with 'diabetes' or related terms found. Looking for any dataset files instead:")
    
    for directory in ['/mnt/data', '/mnt/imported/data']:
        if os.path.exists(directory):
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith(('.csv', '.xlsx', '.parquet')):
                        data_file = os.path.join(root, file)
                        print(f"\nFound data file: {data_file}")
                        
                        try:
                            # Load the data
                            if data_file.endswith('.csv'):
                                df = pd.read_csv(data_file)
                            elif data_file.endswith('.xlsx'):
                                df = pd.read_excel(data_file)
                            elif data_file.endswith('.parquet'):
                                df = pd.read_parquet(data_file)
                                
                            print(f"Shape: {df.shape}")
                            print("Columns:", df.columns.tolist())
                            print("Sample data:")
                            print(df.head(3).to_string())
                            print("---")
                        except Exception as e:
                            print(f"Error reading {data_file}: {str(e)}") 