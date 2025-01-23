import pandas as pd
import os
from sklearn.svm import SVC
import pickle

import yaml

param = yaml.safe_load(open('params.yaml','r'))['model_building']

# Define file paths
base_path = r'D:\ML_ops\pratik\machine_fault\data\preprocessed_data'
train_data_path = os.path.join(base_path, 'train_preprocess.csv')
test_data_path = os.path.join(base_path, 'test_preprocess.csv')

# Load preprocessed data
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

# Split features and target
x_train = train_data.drop(['fail'], axis=1)
y_train = train_data['fail'].astype(int)  # Ensure target is an integer
x_test = test_data.drop(['fail'], axis=1)
y_test = test_data['fail'].astype(int)  # Ensure target is an integer

# Train the SVM model
svc_model = SVC(kernel=param['kernel'],gamma=param['gamma'], C=param['C'], random_state=param['random_state'])
svc_model.fit(x_train, y_train)

# Save the trained model
with open('model.pkl', 'wb') as model_file:
    pickle.dump(svc_model, model_file)

print("Model training complete. Model saved as 'model.pkl'.")