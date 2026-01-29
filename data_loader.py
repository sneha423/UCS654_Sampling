"""
Module for loading, preprocessing, and balancing the credit card dataset
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    def __init__(self, filepath):
        """Initialize DataLoader with dataset path"""
        self.filepath = filepath
        self.df = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load dataset from CSV"""
        print("[*] Loading dataset...")
        self.df = pd.read_csv(self.filepath)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Missing values:\n{self.df.isnull().sum().sum()}")
        return self.df
    
    def check_imbalance(self):
        """Check class distribution"""
        if self.df is None:
            self.load_data()
        
        print("\n[*] Class Distribution (Original):")
        class_dist = self.df.iloc[:, -1].value_counts()
        print(class_dist)
        print(f"Imbalance Ratio: {class_dist.max() / class_dist.min():.2f}:1")
        return class_dist
    
    def preprocess(self):
        """
        Preprocess data:
        - Drop unnecessary columns
        - Separate features and labels
        - Perform train-test split
        - Apply SMOTE for balancing
        - Scale features
        """
        print("\n[*] Preprocessing data...")
        
        if self.df is None:
            self.load_data()
        
        # Assuming last column is target
        X = self.df.iloc[:, :-1]
        y = self.df.iloc[:, -1]
        
        # Train-test split (stratified to maintain class distribution)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        print(f"\nClass distribution in training set (before SMOTE):")
        print(y_train.value_counts())
        
        # Apply SMOTE on training data only
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        print(f"\nClass distribution in training set (after SMOTE):")
        print(pd.Series(y_train_balanced).value_counts())
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_balanced)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.X_train_scaled = X_train_scaled
        self.X_test_scaled = X_test_scaled
        self.y_train = y_train_balanced
        self.y_test = y_test
        
        print("\n[✓] Data preprocessing complete")
        return X_train_scaled, X_test_scaled, y_train_balanced, y_test
    
    def get_processed_data(self):
        """Return preprocessed data"""
        if self.X_train_scaled is None:
            self.preprocess()
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test


# Example usage
if __name__ == "__main__":
    loader = DataLoader("Creditcard_data.csv")
    loader.load_data()
    loader.check_imbalance()
    X_train, X_test, y_train, y_test = loader.preprocess()