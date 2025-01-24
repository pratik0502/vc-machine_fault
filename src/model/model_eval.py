import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import os

# Define file paths
base_path = r'D:\ML_ops\pratik\machine_fault_cc\machine_fault_prediction\data\preprocessed_data'
test_data_path = os.path.join(base_path, 'test_preprocess.csv')

# Load test data
test_data = pd.read_csv(test_data_path)
x_test = test_data.drop(['fail'], axis=1)
y_test = test_data['fail'].astype(int)

# Load the trained model
svc = pickle.load(open('model.pkl', 'rb'))

# Make predictions
y_pred = svc.predict(x_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

# Store metrics in a dictionary
metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1
}

# Convert metrics to JSON
metrics_json = json.dumps(metrics, indent=4)

# Save metrics to a file
with open("metrics.json", "w") as f:
    f.write(metrics_json)

print("Metrics saved to 'metrics.json'")
