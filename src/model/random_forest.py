

from sklearn.ensemble import RandomForestClassifier


def train_random_forest(X_train, y_train, params=None):
    """
    Train a Random Forest model.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        params (dict): Hyperparameters for the Random Forest model.
        
    Returns:
        model: Trained Random Forest model.
    """
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': None,
            'random_state': 42
        }
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model