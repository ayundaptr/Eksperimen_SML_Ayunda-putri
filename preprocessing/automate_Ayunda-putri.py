# automate_Ayunda-putri.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import datetime  # untuk timestamp file

class HousingPreprocessor:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.df = None
        self.num_cols = None
        self.cat_cols = None
        self.scaler = StandardScaler()
        self.le = LabelEncoder()

    # =======================
    # Data Loading
    # =======================
    def load_dataset(self):
        """Membaca dataset dari CSV"""
        self.df = pd.read_csv(self.dataset_path)
        print("Dataset loaded. Shape:", self.df.shape)
        return self.df

    # =======================
    # Cleaning
    # =======================
    def remove_duplicates(self):
        """Hapus duplikat"""
        before = self.df.shape[0]
        self.df = self.df.drop_duplicates()
        after = self.df.shape[0]
        print(f"Removed {before - after} duplicate rows.")
        return self.df

    def handle_missing_values(self):
        """Isi missing values"""
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
        """Deteksi dan hapus outlier menggunakan IQR"""
        for col in self.num_cols:
            if col in ["bedrooms","bathrooms","stories","parking"]:
                continue  # jangan ubah kolom penting
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            self.df = self.df[(self.df[col] >= lower) & (self.df[col] <= upper)]
        print("Outliers handled.")
        return self.df

    # =======================
    # Feature Engineering
    # =======================
    def scale_numeric(self):
        """Scaling fitur numerik kecuali kolom penting"""
        cols_to_scale = [col for col in self.num_cols if col not in ["bedrooms","bathrooms","stories","parking"]]
        self.df["price_original"] = self.df["price"]  # simpan harga asli
        self.df[cols_to_scale] = self.scaler.fit_transform(self.df[cols_to_scale])
        print("Numeric features scaled.")
        return self.df

    def encode_categorical(self):
        """Encoding fitur kategorikal"""
        for col in self.cat_cols:
            self.df[col] = self.le.fit_transform(self.df[col])
        print("Categorical features encoded.")
        return self.df

    def bin_price(self):
        """Binning harga menjadi Low, Medium, High"""
        self.df['price_bin'] = pd.qcut(self.df['price_original'], q=3, labels=["Low","Medium","High"])
        print("Price binned into categories.")
        return self.df

    # =======================
    # Full Preprocessing
    # =======================
    def preprocess_all(self):
        self.load_dataset()
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
    dataset_path = "housing_raw.csv"  # path relatif ke root repo
    preprocessor = HousingPreprocessor(dataset_path)
    df_processed = preprocessor.preprocess_all()

    # Output folder
    output_dir = "preprocessing/housing_preprocessed"
    os.makedirs(output_dir, exist_ok=True)

    # Gunakan timestamp agar setiap file unik
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"housing_automate_preprocessed_{timestamp}.csv")

    df_processed.to_csv(output_file, index=False)
    print(f"Preprocessed dataset saved as '{output_file}'")

if __name__ == "__main__":
    main()