import os
import pandas as pd
from pathlib import Path
from typing import Union

def load_creditcard_csv(path: Union[str, os.PathLike] = 'data/creditcard.csv') -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Please download 'creditcard.csv' from https://www.kaggle.com/mlg-ulb/creditcardfraud and place it at {path}")
    
    # try different encodings
    encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(path, encoding=encoding)
            if 'Class' not in df.columns:
                raise ValueError("Expected a 'Class' column with 0/1 labels.")
            print(f"Successfully loaded CSV with {encoding} encoding")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error with {encoding} encoding: {e}")
            continue
    
    raise ValueError("Could not read CSV file with any of the attempted encodings: utf-8, latin1, cp1252, iso-8859-1")

def load_nsl_kdd_data(train_path: Union[str, os.PathLike] = 'data/NSL-KDD dataset/KDDTrain+.txt',
                     test_path: Union[str, os.PathLike] = 'data/NSL-KDD dataset/KDDTest+.txt') -> pd.DataFrame:
    """Load NSL-KDD dataset and combine train/test sets"""
    train_path = Path(train_path)
    test_path = Path(test_path)
    
    if not train_path.exists():
        raise FileNotFoundError(f"NSL-KDD training data not found at {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"NSL-KDD test data not found at {test_path}")
    
    # Define column names for NSL-KDD dataset
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'difficulty'
    ]
    
    print("Loading NSL-KDD training data...")
    train_df = pd.read_csv(train_path, names=columns, header=None)
    print(f"Loaded {len(train_df)} training samples")
    
    print("Loading NSL-KDD test data...")
    test_df = pd.read_csv(test_path, names=columns, header=None)
    print(f"Loaded {len(test_df)} test samples")
    
    # Combine train and test data
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    print(f"Combined dataset: {len(combined_df)} total samples")
    
    # Convert attack_type to binary classification (normal vs attack)
    # Normal = 0, Any attack = 1
    combined_df['Class'] = (combined_df['attack_type'] != 'normal').astype(int)
    
    # Remove the original attack_type and difficulty columns
    combined_df = combined_df.drop(columns=['attack_type', 'difficulty'])
    
    print(f"Binary classification: {combined_df['Class'].sum()} attacks, {len(combined_df) - combined_df['Class'].sum()} normal")
    print(f"Attack ratio: {combined_df['Class'].mean():.4f}")
    
    return combined_df
