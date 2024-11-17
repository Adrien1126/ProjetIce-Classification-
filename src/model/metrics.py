from sklearn.metrics import classification_report, roc_auc_score


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.
        
    Returns:
        dict: Evaluation metrics (classification report and ROC-AUC score).
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_proba)
    return {
        "classification_report": report,
        "roc_auc": auc
    }
