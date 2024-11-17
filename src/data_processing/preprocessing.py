import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_data(features_path, targets_path):
    """
    Charger les données depuis les fichiers CSV.
    """
    features = pd.read_csv(features_path)
    targets = pd.read_csv(targets_path)
    return features, targets

# Display basic information about the datasets
def describe_data(features, targets):
    print("Features Dataset Info:")
    print(features.info())
    print("\nBasic Statistics for Features:")
    print(features.describe())

    print("\nChecking for Missing Values in Features:")
    print(features.isnull().sum())

    print("\nTarget Dataset Info:")
    print(targets.info())
    print("\nBasic Statistics for Targets:")
    print(targets.describe())

    print("\nChecking for Missing Values in Targets:")
    print(targets.isnull().sum())

    print("\nFirst few rows of Features:")
    print(features.head())

    print("\nFirst few rows of Targets:")
    print(targets.head())


def transform_target_to_binary(target, threshold):
    """
    Transformer une cible quantitative en variable binaire en utilisant un seuil.
    """
    return (target > threshold).astype(int)

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

def preprocess_features(features):
    """
    Preprocesses the features DataFrame by performing the following steps:
    1. Drops the 'time' column.
    2. Applies scaling (standardization) to 'u10', 'v10', and 't2m'.
    3. Applies logarithmic transformation to 'SST' and removes the original column.
    4. Normalizes 'SIC' to a [0, 1] range.

    Args:
        features (pd.DataFrame): DataFrame containing the feature set.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    # 1. Drop the 'time' column
    if 'time' in features.columns:
        features = features.drop(columns=['time'])

    # 2. Scaling (standardization) for 'u10', 'v10', and 't2m'
    scaler = StandardScaler()
    if {'u10', 'v10', 't2m'}.issubset(features.columns):
        features[['u10', 'v10', 't2m']] = scaler.fit_transform(features[['u10', 'v10', 't2m']])

    # 3. Logarithmic transformation for 'SST'
    if 'SST' in features.columns:
        min_value = features['SST'].min()
        if min_value < 0:
            features['SST_log'] = np.log1p(features['SST'] - min_value + 1)
        else:
            features['SST_log'] = np.log1p(features['SST'])
        features = features.drop(columns=['SST'])

    # 4. Normalize 'SIC'
    if 'SIC' in features.columns:
        sic_scaler = MinMaxScaler()
        features['SIC_scaled'] = sic_scaler.fit_transform(features[['SIC']])
        features = features.drop(columns=['SIC'])

    return features


def split_data(features, target, test_size=0.15, random_state=42):
    """
    Splits the dataset into training, validation (optional), and test sets.

    Args:
        features (pd.DataFrame): Feature dataset.
        target (pd.Series): Target variable.
        test_size (float): Proportion of the dataset to include in the test split.
        val_size (float): Proportion of the training set to include in the validation split.
        random_state (int): Random seed for reproducibility.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test: Split datasets.
    """
    # Split into training + validation and test
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state, stratify=target
    )

    return X_train, X_test, y_train, y_test
