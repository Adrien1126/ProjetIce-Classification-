from sklearn.ensemble import AdaBoostClassifier


def train_adaboost(X_train, y_train, params=None):
    """
    Train an AdaBoost model.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        params (dict): Hyperparameters for the AdaBoost model.
        
    Returns:
        model: Trained AdaBoost model.
    """
    if params is None:
        params = {
            'n_estimators': 50,
            'learning_rate': 1.0,
            'random_state': 42
        }
    model = AdaBoostClassifier(**params)
    model.fit(X_train, y_train)
    return model