from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

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

def plot_roc_curve(model,model_name, X_test, y_test):
    """
    Plot the ROC curve for a given model's predictions.
    
    Args:
        model: Trained model.
        model_name (str): Name of the model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.
    """
    y_proba = model.predict_proba(X_test)[:, 1]

    # Calculate ROC curve metrics
    fpr, tpr, thresholds  = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)  # Calculate the AUC score

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", color='blue', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', label="Random guess", lw=2)  # Diagonal reference line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()