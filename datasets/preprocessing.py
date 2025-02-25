import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import argparse
import os

class DataPreprocessor:
    def __init__(self, input_file, output_file=None, normalize=True, encode_labels=True):
        """
        Initializes the DataPreprocessor.

        Args:
            input_file (str): Path to the raw dataset file.
            output_file (str, optional): Path to save the processed dataset.
            normalize (bool): Whether to normalize numerical features.
            encode_labels (bool): Whether to encode categorical labels.
        """
        self.input_file = input_file
        self.output_file = output_file
        self.normalize = normalize
        self.encode_labels = encode_labels
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def load_data(self):
        """Loads the dataset into a Pandas DataFrame."""
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"Dataset file '{self.input_file}' not found.")
        
        print(f"Loading dataset from {self.input_file}...")
        df = pd.read_csv(self.input_file)
        return df

    def clean_data(self, df):
        """Handles missing values and removes duplicates."""
        print("Cleaning data: handling missing values and removing duplicates...")
        
        # Drop columns with excessive missing values (optional)
        df.dropna(thresh=len(df) * 0.5, axis=1, inplace=True)

        # Fill missing values with median for numerical features
        for col in df.select_dtypes(include=['number']).columns:
            df[col].fillna(df[col].median(), inplace=True)
        
        # Fill missing values with mode for categorical features
        for col in df.select_dtypes(include=['object']).columns:
            df[col].fillna(df[col].mode()[0], inplace=True)

        # Remove duplicate rows
        df.drop_duplicates(inplace=True)

        return df

    def feature_engineering(self, df):
        """Extracts useful network flow features and converts categorical data."""
        print("Performing feature extraction and engineering...")
        
        # Convert timestamp to useful time-based features
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df.drop(columns=['timestamp'], inplace=True)

        # Encode protocol types if available
        if 'protocol' in df.columns:
            df['protocol'] = df['protocol'].astype('category').cat.codes

        # Extract source-destination IP pair as unique flow identifier
        if {'src_ip', 'dst_ip'}.issubset(df.columns):
            df['flow_id'] = df['src_ip'] + "_" + df['dst_ip']
            df.drop(columns=['src_ip', 'dst_ip'], inplace=True)

        return df

    def normalize_features(self, df):
        """Standardizes numerical features using StandardScaler."""
        print("Normalizing numerical features...")
        
        num_cols = df.select_dtypes(include=['number']).columns
        df[num_cols] = self.scaler.fit_transform(df[num_cols])

        return df

    def encode_labels(self, df):
        """Encodes the target labels (benign/malicious) if applicable."""
        if 'label' in df.columns:
            print("Encoding labels...")
            df['label'] = self.label_encoder.fit_transform(df['label'])
        return df

    def process(self):
        """Executes the full preprocessing pipeline."""
        df = self.load_data()
        df = self.clean_data()
        df = self.feature_engineering(df)

        if self.normalize:
            df = self.normalize_features(df)
        if self.encode_labels:
            df = self.encode_labels(df)

        # Save processed dataset if output path is provided
        if self.output_file:
            df.to_csv(self.output_file, index=False)
            print(f"Processed dataset saved to {self.output_file}")
        
        return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess network traffic dataset for MGAN.")
    parser.add_argument("--input", type=str, required=True, help="Path to the raw dataset CSV file")
    parser.add_argument("--output", type=str, required=False, help="Path to save the processed dataset")
    parser.add_argument("--normalize", action="store_true", help="Apply feature normalization")
    parser.add_argument("--encode_labels", action="store_true", help="Apply label encoding")

    args = parser.parse_args()

    preprocessor = DataPreprocessor(input_file=args.input, output_file=args.output,
                                    normalize=args.normalize, encode_labels=args.encode_labels)
    processed_data = preprocessor.process()
