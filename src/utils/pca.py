from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

def apply_pca_to_mar_features(features, mar_columns, n_components=1):
    """
    Applies PCA to reduce the dimensions of MAR-related features.

    Args:
        features (pd.DataFrame): DataFrame containing the features.
        mar_columns (list): List of MAR-related feature names (e.g., ['r1_MAR', 'r2_MAR', ...]).
        n_components (int): Number of principal components to keep. Default is 1.

    Returns:
        pd.DataFrame: Updated DataFrame with the new PCA feature added.
        float: Explained variance ratio for the first principal component.
    """
    # 1. Standardization
    scaler = StandardScaler()
    features_standardized = scaler.fit_transform(features[mar_columns])

    # 2. PCA
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(features_standardized)

    # 3. Add the PCA component to the DataFrame
    features['r_MAR_pca'] = pca_features[:, 0]

    # 4. Return updated features and explained variance ratio
    explained_variance = pca.explained_variance_ratio_[0]
    return features, explained_variance
