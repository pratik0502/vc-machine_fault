import pandas as pd
import os
import yaml

from sklearn.model_selection import train_test_split

def load_param(params: str) -> float: 

    test_size = yaml.safe_load(open(params,'r'))['data_ingestion']['test_size'] 

    return test_size   

def load_data(data: str) -> pd.DataFrame:

    data = pd.read_csv(data)

    return data

def save_data(data_path: str,train_data: pd.DataFrame,train_data_path: str,test_data: pd.DataFrame,test_data_path: str) -> None:

    # Path to the directory where the CSV will be saved
    output_dir = data_path

    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)

    train_data.to_csv(os.path.join(train_data_path,'train.csv'))

    test_data.to_csv(os.path.join(test_data_path,'test.csv'))

def main()-> None:
    
    test_size = load_param('params.yaml')

    data = load_data(r"D:\ML_ops\archive\data.csv")

    train_data,test_data = train_test_split(data,test_size=test_size,random_state=42)

    save_data( r"D:\ML_ops\pratik\machine_fault_cc\machine_fault_prediction\data\raw",train_data,r'D:\ML_ops\pratik\machine_fault_cc\machine_fault_prediction\data\raw',test_data,r'D:\ML_ops\pratik\machine_fault_cc\machine_fault_prediction\data\raw')

    




if __name__ == '__main__':
    main()