from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

NUMERIC_SCALE_CREDITCARD = ['Time', 'Amount']
NUMERIC_SCALE_NSLKDD = [
    'duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
    'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
]
CATEGORICAL_COLS_NSLKDD = ['protocol_type', 'service', 'flag']

def preprocess_df(df: pd.DataFrame, dataset_type: str = 'creditcard'):
    df = df.copy()
    
    if dataset_type == 'creditcard':
        # Credit card fraud dataset preprocessing
        scale_cols = NUMERIC_SCALE_CREDITCARD
        for col in scale_cols:
            if col in df.columns:
                scaler = StandardScaler()
                df[col] = scaler.fit_transform(df[[col]])
    
    elif dataset_type == 'nsl_kdd':
        # NSL-KDD dataset preprocessing
        # Encode categorical variables
        label_encoders = {}
        for col in CATEGORICAL_COLS_NSLKDD:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le
        
        # Scale numeric variables
        for col in NUMERIC_SCALE_NSLKDD:
            if col in df.columns:
                scaler = StandardScaler()
                df[col] = scaler.fit_transform(df[[col]])
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    X = df.drop(columns=['Class'])
    y = df['Class'].astype(int).values
    return X, y
