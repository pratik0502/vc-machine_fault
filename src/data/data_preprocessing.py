import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler,MinMaxScaler

# Load raw data
train_data = pd.read_csv(r'D:\ML_ops\pratik\machine_fault_cc\machine_fault_prediction\data\raw\train.csv')
test_data = pd.read_csv(r'D:\ML_ops\pratik\machine_fault_cc\machine_fault_prediction\data\raw\test.csv')

# Standardize data
# scaler = StandardScaler()
scaler = MinMaxScaler()
train_preprocess = scaler.fit_transform(train_data)
test_preprocess = scaler.transform(test_data)

# Convert back to DataFrame
train_preprocess_df = pd.DataFrame(train_preprocess, columns=train_data.columns)
test_preprocess_df = pd.DataFrame(test_preprocess, columns=test_data.columns)

# Define the output directory
data_path = os.path.join("data", "preprocessed_data")

# Ensure the directory exists
os.makedirs(data_path, exist_ok=True)

# Save preprocessed data
train_preprocess_df.to_csv(os.path.join(data_path, 'train_preprocess.csv'), index=False)
test_preprocess_df.to_csv(os.path.join(data_path, 'test_preprocess.csv'), index=False)

print(f"Preprocessed data saved to: {data_path}")