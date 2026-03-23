import pandas as pd
import numpy as np
import os
import ssl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

class CreditDataAnalyzer:
    def __init__(self, data_dir="data/raw", filename="default_of_credit_card_clients.csv"):
        self.data_dir = data_dir
        self.file_path = os.path.join(self.data_dir, filename)
        self.df = None
        
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        """Downloads the dataset if it doesn't exist locally, or loads it from disk."""
        # Disable SSL verification to allow downloading
        ssl._create_default_https_context = ssl._create_unverified_context
        os.makedirs(self.data_dir, exist_ok=True)
        
  
        if not os.path.exists(self.file_path):
            print("Downloading dataset from UCI...")
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
            
            self.df = pd.read_excel(url, header=1)
            self.df.to_csv(self.file_path, index=False)
            print("✅ Dataset downloaded and converted to CSV in data/raw/")
        else:
            print("✅ CSV dataset already exists locally.")
            self.df = pd.read_csv(self.file_path)
            
        return self.df

    def explore_basics(self):
        """Displays basic information about the dataset."""
        print("\n--- Basic Exploration ---")
        print(f"Dataset shape: {self.df.shape}")
        print("\nColumns:", self.df.columns.tolist())
        print("\nMissing values per column:")
        print(self.df.isnull().sum())
        print("\nData preview:")



    def clean_data(self):
        """Cleans duplicate rows, removes the ID column, and renames the target variable."""
        print("\n--- Data Cleaning ---")
        
        # Handle duplicates
        duplicate_count = self.df.duplicated().sum()
        print(f"Number of duplicate rows: {duplicate_count}")
        if duplicate_count > 0:
            self.df.drop_duplicates(inplace=True)
            print(f"Duplicates removed. New dataset shape: {self.df.shape}")
            
        # Remove the ID column as it's not predictive
        if 'ID' in self.df.columns:
            self.df.drop(columns=['ID'], inplace=True)
            
        # Rename the target variable for simplicity
        # (Handles both common formats found in this dataset)
        target_names = ['default.payment.next.month', 'default payment next month']
        for name in target_names:
            if name in self.df.columns:
                self.df.rename(columns={name: 'IsDefault'}, inplace=True)
                break
                
        print("Data cleaned (ID removed, target renamed to 'IsDefault').")


    def plot_target_distribution(self):
        """Plots the distribution of the target variable."""
        target_counts = self.df['IsDefault'].value_counts(normalize=True) * 100
        print("\nTarget distribution (%) :\n", target_counts)
        
        self.df['IsDefault'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
        plt.title("Distribution of 'Default payment next month'")
        plt.ylabel("Number of clients")
        plt.xticks(rotation=0)
        plt.show()

    def check_bias(self, feature='SEX'):
        """Checks for potential bias across a specific categorical feature."""
        print(f"\n--- Bias check for feature '{feature}' ---")
        bias_check = pd.crosstab(self.df[feature], self.df['IsDefault'], normalize='index')
        print(bias_check)
        
        bias_check.plot(kind='bar', stacked=True, color=['skyblue', 'salmon'])
        plt.title(f"Payment Default by {feature}")
        plt.ylabel("Proportion")
        plt.xticks(rotation=0)
        plt.legend(title='IsDefault')
        plt.show()

    def plot_correlation_matrix(self):
        """BONUS: Plots a heatmap of linear correlations between features."""
        plt.figure(figsize=(16, 10))
        # Compute correlation matrix
        corr = self.df.corr()
        # Mask the upper triangle for better readability
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=False, fmt=".2f", linewidths=.5)
        plt.title("Feature Correlation Matrix")
        plt.show()

    def plot_feature_boxplot(self, features):
        """BONUS: Plots boxplots for specific features to detect outliers."""
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.df[features], palette="Set2")
        plt.title("Outlier Detection (Boxplots)")
        plt.xticks(rotation=45)
        plt.show()

    def prepare_for_modeling(self, test_size=0.2, random_state=42):
        """Splits the data into Train/Test sets and applies standardization."""
        print("\n--- Machine Learning Preparation ---")
        X = self.df.drop(columns=['IsDefault'])
        y = self.df['IsDefault']
        
        # Split data (stratified to maintain target distribution)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Standardization (Scaling)
        scaler = StandardScaler()
        
        # Fit on train data only to prevent data leakage, transform both train and test
        self.X_train_scaled = pd.DataFrame(scaler.fit_transform(self.X_train), 
                                           columns=X.columns, index=self.X_train.index)
        
        self.X_test_scaled = pd.DataFrame(scaler.transform(self.X_test), 
                                          columns=X.columns, index=self.X_test.index)
        
        print("Data split and standardized successfully!")
        print(f"Training set size: {self.X_train_scaled.shape}")
        print(f"Test set size: {self.X_test_scaled.shape}")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test

    def compute_model_weights(self):
        """Computes class weights to handle dataset imbalance."""
        if self.y_train is None:
            print("Please run 'prepare_for_modeling()' first before computing weights.")
            return
            
        weights = compute_class_weight('balanced', classes=np.unique(self.y_train), y=self.y_train)
        print(f"\n--- Recommended weights to handle class imbalance ---")
        print(f"Class 0 -> {weights[0]:.2f}")
        print(f"Class 1 -> {weights[1]:.2f}")
        return weights