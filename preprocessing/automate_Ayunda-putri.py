# =======================
# 1. Import Library
# =======================
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import os
import datetime

# =======================
# 2. Automated Preprocessor
# =======================
class WaterQualityAutoPreprocessor:
    def __init__(self, dataset_path, target_col="is_safe"):
        self.dataset_path = dataset_path
        self.target_col = target_col
        self.df = None
        self.num_cols = None
        self.scaler = StandardScaler()

    # Load Dataset
    def load_dataset(self):
        self.df = pd.read_csv(self.dataset_path)
        print("Dataset loaded:", self.df.shape)
        return self.df

    # Basic EDA
    def basic_eda(self):
        print("\n--- INFO ---")
        print(self.df.info())
        print("\n--- DESCRIBE ---")
        print(self.df.describe())
        print("\n--- MISSING VALUES ---")
        print(self.df.isnull().sum())
        return self.df

    # Convert to numeric
    def convert_to_numeric(self):
        for col in self.df.columns:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
        print("Non-numeric values converted to NaN.")
        return self.df

    # Remove Duplicates
    def remove_duplicates(self):
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        after = len(self.df)
        print(f"Duplicates removed: {before - after}")
        return self.df

    # Handle Missing Values (median)
    def handle_missing_values(self):
        self.num_cols = self.df.columns
        self.df[self.num_cols] = self.df[self.num_cols].apply(lambda x: x.fillna(x.median()))
        print("Missing values filled with median.")
        return self.df

    # Handle Outliers (IQR capping)
    def handle_outliers(self):
        feature_cols = [c for c in self.num_cols if c != self.target_col]
        for col in feature_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            self.df[col] = np.clip(self.df[col], lower, upper)
        print("Outliers handled using IQR capping.")
        return self.df

    # Standardize Features
    def standardize_features(self):
        feature_cols = [c for c in self.num_cols if c != self.target_col]
        self.df[feature_cols] = self.scaler.fit_transform(self.df[feature_cols])
        print("Features standardized using StandardScaler.")
        return self.df

    # Full Pipeline
    def preprocess_all(self):
        self.load_dataset()
        self.basic_eda()
        self.convert_to_numeric()
        self.remove_duplicates()
        self.handle_missing_values()
        self.handle_outliers()
        self.standardize_features()
        print("PREPROCESSING COMPLETE.")
        return self.df

# =======================
# 3. Main Function
# =======================
def main():
    dataset_path = "waterquality.csv"  # sesuaikan path dataset
    output_dir = "preprocessing/waterquality_preprocessed"

    preprocessor = WaterQualityAutoPreprocessor(dataset_path)
    df_processed = preprocessor.preprocess_all()

    # Buat folder output
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"waterquality_automate_preprocessed_{timestamp}.csv")
    df_processed.to_csv(output_file, index=False)
    print(f"Saved to: {output_file}")

    # Visualisasi Korelasi
    feature_cols = [c for c in df_processed.columns if c != preprocessor.target_col]
    plt.figure(figsize=(12,10))
    sns.heatmap(df_processed[feature_cols].corr(), cmap="coolwarm")
    plt.title("Correlation Heatmap (After Preprocessing)")
    plt.show()

if __name__ == "__main__":
    main()