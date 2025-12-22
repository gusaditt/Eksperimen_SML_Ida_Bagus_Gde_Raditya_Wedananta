import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def preprocess_data(
    input_path="dataset_raw/diabetes.csv",
    output_path="preprocessing/dataset_preprocessing/dataset_preprocessing.csv"
):
    # Load dataset
    df = pd.read_csv(input_path)

    # Kolom dengan nilai 0 tidak realistis
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    df_clean = df.copy()
    for col in zero_cols:
        df_clean[col] = df_clean[col].replace(0, np.nan)

    # Handle missing value
    df_clean.fillna(df_clean.median(), inplace=True)

    # Pisahkan fitur dan target
    X = df_clean.drop("Outcome", axis=1)
    y = df_clean["Outcome"]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # Gabungkan kembali
    df_preprocessed = X_scaled.copy()
    df_preprocessed["Outcome"] = y.values

    # Pastikan folder output ada
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Simpan hasil
    df_preprocessed.to_csv(output_path, index=False)

    print("Preprocessing selesai. Dataset disimpan di:", output_path)


if __name__ == "__main__":
    preprocess_data()
