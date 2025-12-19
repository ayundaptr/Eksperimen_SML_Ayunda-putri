# automate_Ayunda-putri.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import datetime

class HousingPreprocessor:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.df = None
        self.num_cols = None
        self.cat_cols = None
        self.scaler = StandardScaler()
        self.le = LabelEncoder()

    # =======================
    # 1. Memuat Dataset
    # =======================
    def load_dataset(self):
        """Membaca dataset dari CSV"""
        self.df = pd.read_csv(self.dataset_path)
        print("Dataset loaded. Shape:", self.df.shape)
        print(self.df.head())
        return self.df

    # =======================
    # 2. Exploratory Data Analysis Sederhana
    # =======================
    def basic_eda(self):
        print("\n--- Data Info ---")
        print(self.df.info())
        print("\n--- Descriptive Statistics ---")
        print(self.df.describe())
        print("\n--- Missing Values ---")
        print(self.df.isnull().sum())
        return self.df

    # =======================
    # 3. Data Preprocessing
    # =======================
    def remove_duplicates(self):
        before = self.df.shape[0]
        self.df = self.df.drop_duplicates()
        after = self.df.shape[0]
        print(f"Removed {before - after} duplicate rows.")
        return self.df

    def handle_missing_values(self):
        self.num_cols = self.df.select_dtypes(include=['int64','float64']).columns
        self.cat_cols = self.df.select_dtypes(include='object').columns

        # Numerik: isi dengan median
        self.df[self.num_cols] = self.df[self.num_cols].fillna(self.df[self.num_cols].median())

        # Kategorikal: isi dengan modus
        for col in self.cat_cols:
            self.df[col] = self.df[col].fillna(self.df[col].mode()[0])

        print("Missing values handled.")
        return self.df

    def handle_outliers(self):
        for col in self.num_cols:
            # Bisa sesuaikan kolom penting jika ada
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            self.df = self.df[(self.df[col] >= lower) & (self.df[col] <= upper)]
        print("Outliers handled.")
        return self.df

    def scale_numeric(self):
        cols_to_scale = [col for col in self.num_cols if col not in ["price"]]
        if "price" in self.df.columns:
            self.df["price_original"] = self.df["price"]
        self.df[cols_to_scale] = self.scaler.fit_transform(self.df[cols_to_scale])
        print("Numeric features scaled.")
        return self.df

    def encode_categorical(self):
        for col in self.cat_cols:
            self.df[col] = self.le.fit_transform(self.df[col])
        print("Categorical features encoded.")
        return self.df

    def bin_price(self):
        if "price_original" in self.df.columns:
            self.df['price_bin'] = pd.qcut(self.df['price_original'], q=3, labels=["Low","Medium","High"])
            print("Price binned into categories.")
        return self.df

    # =======================
    # 4. Full Preprocessing
    # =======================
    def preprocess_all(self):
        self.load_dataset()
        self.basic_eda()
        self.remove_duplicates()
        self.handle_missing_values()
        self.handle_outliers()
        self.scale_numeric()
        self.encode_categorical()
        self.bin_price()
        print("Preprocessing complete. Dataset ready for training.")
        return self.df

# =======================
# Main function
# =======================
def main():
    dataset_path = "housing_raw.csv"  # ganti sesuai dataset
    preprocessor = HousingPreprocessor(dataset_path)
    df_processed = preprocessor.preprocess_all()

    # Folder output
    output_dir = "preprocessing/housing_preprocessed"
    os.makedirs(output_dir, exist_ok=True)

    # File output dengan timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"housing_automate_preprocessed_{timestamp}.csv")

    df_processed.to_csv(output_file, index=False)
    print(f"Preprocessed dataset saved as '{output_file}'")

if __name__ == "__main__":
    main()