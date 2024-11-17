import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(features_path, targets_path):
    """
    Charger les données depuis les fichiers CSV.
    """
    features = pd.read_csv(features_path)
    targets = pd.read_csv(targets_path)
    return features, targets

def transform_target_to_binary(target, threshold):
    """
    Transformer une cible quantitative en variable binaire en utilisant un seuil.
    """
    return (target > threshold).astype(int)

def preprocess_data(features, target_binary):
    """
    Séparer les données en ensembles d'entraînement et de test et standardiser les caractéristiques.
    """
    X = features
    y = target_binary

    # Séparation des données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test