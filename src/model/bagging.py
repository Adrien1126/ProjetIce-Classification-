
from sklearn.ensemble import BaggingClassifier


def train_bagging(X_train, y_train, params=None):
    """
    Train a Bagging model.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        params (dict): Hyperparameters for the Bagging model.
        
    Returns:
        model: Trained Bagging model.
    """
    if params is None:
        params = {
            'n_estimators': 50,
            'random_state': 42
        }
    model = BaggingClassifier(**params)
    model.fit(X_train, y_train)
    return model