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

    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)

    # Compute ROC-AUC
    auc = roc_auc_score(y_test, y_proba)

    return {
        "classification_report": report,
        "roc_auc": auc
    }


def evaluate_and_print(model, model_name, X_test, y_test):
    """
    Evaluate a model and print the results.

    Args:
        model: Trained model.
        model_name (str): Name of the model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.
    """
    eval_results = evaluate_model(model, X_test, y_test)

    print(f"\n========== Evaluation for {model_name} ==========")
    print(f"{model_name} ROC-AUC: {eval_results['roc_auc']:.4f}")
    print(f"{model_name} Classification Report:")
    for class_label, metrics in eval_results["classification_report"].items():
        if isinstance(metrics, dict):  # Skip 'accuracy', 'macro avg', etc.
            print(f"  Class {class_label} - Precision: {metrics['precision']:.2f}, "
                  f"Recall: {metrics['recall']:.2f}, F1-Score: {metrics['f1-score']:.2f}")
    print("==============================================\n")

