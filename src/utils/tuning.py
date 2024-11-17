from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def optimize_hyperparameters(model, param_grid, X_train, y_train, cv=5, scoring='recall', search_type='grid', n_iter=50):
    """
    Optimizes hyperparameters for a given model using GridSearchCV or RandomizedSearchCV.

    Args:
        model: The machine learning model to optimize.
        param_grid (dict): Dictionary containing hyperparameters to tune.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        cv (int): Number of cross-validation folds. Default is 5.
        scoring (str): Scoring metric for optimization. Default is 'recall'.
        search_type (str): Type of search, either 'grid' or 'random'. Default is 'grid'.
        n_iter (int): Number of iterations for RandomizedSearchCV. Default is 50 (only used if search_type='random').

    Returns:
        best_model: Model with the best hyperparameters.
        best_params: Dictionary of the best hyperparameters.
    """
    if search_type == 'grid':
        search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, verbose=2, n_jobs=-1)
    elif search_type == 'random':
        search = RandomizedSearchCV(model, param_grid, cv=cv, scoring=scoring, n_iter=n_iter, verbose=2, n_jobs=-1)
    else:
        raise ValueError("search_type must be either 'grid' or 'random'")
    
    # Fit the search
    search.fit(X_train, y_train)
    
    return search.best_estimator_, search.best_params_