# automate_NamaAnda.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

def preprocess_telco(input_path, output_path):
    # Load dataset
    df = pd.read_csv(input_path)
    
    # Drop customerID
    df.drop('customerID', axis=1, inplace=True)

    # Handle TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)

    # Encode target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Encode categorical features
    cat_cols = df.select_dtypes(include='object').columns
    df[cat_cols] = df[cat_cols].apply(LabelEncoder().fit_transform)

    # Scale numeric columns
    scaler = StandardScaler()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Simpan hasil preprocessing
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Data preprocessed disimpan di: {output_path}")

if __name__ == "__main__":
    input_csv = '../WA_Fn-UseC_-Telco-Customer-Churn.csv'
    output_csv = "./clean_telco.csv"
    preprocess_telco(input_csv, output_csv)
