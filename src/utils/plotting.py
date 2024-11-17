import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_target_distribution(targets, binary_threshold=None):
    """
    Plot the distribution of the target variable.
    
    Args:
        targets (pd.Series): Target variable.
        binary_threshold (int or float, optional): Threshold for binary transformation.
    """
    plt.figure(figsize=(6, 4))
    sns.histplot(targets, bins=30, kde=False, edgecolor="black", color="skyblue")
    plt.title('Distribution of Y1 (Target)')
    plt.xlabel('Y1')
    plt.ylabel('Count')
    plt.show()

    if binary_threshold is not None:
        binary_target = (targets >= binary_threshold).astype(int)
        plt.figure(figsize=(6, 4))
        sns.countplot(x=binary_target, palette="viridis")
        plt.title(f'Binary Target Distribution (Threshold: {binary_threshold})')
        plt.xlabel('Y1 Binary')
        plt.ylabel('Count')
        plt.xticks([0, 1], ['Low', 'High'])
        plt.show()


def plot_correlation_heatmap(features):
    """
    Plot the correlation heatmap for features.
    
    Args:
        features (pd.DataFrame): Feature dataset.
    """
    plt.figure(figsize=(12, 10))
    correlation_matrix = features.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix of Features")
    plt.show()


def plot_feature_target_correlation(features, targets):
    """
    Plot correlation of features with the target variable.
    
    Args:
        features (pd.DataFrame): Feature dataset.
        targets (pd.Series): Target variable.
    """
    correlations = features.corrwith(targets)
    plt.figure(figsize=(8, 6))
    correlations.sort_values().plot(kind='barh', color='teal')
    plt.title('Correlation of Features with Y1')
    plt.xlabel('Correlation')
    plt.ylabel('Features')
    plt.show()


def plot_feature_distributions(features):
    """
    Plot histograms of all features to check distributions.
    
    Args:
        features (pd.DataFrame): Feature dataset.
    """
    features.hist(figsize=(14, 12), bins=20, color='blue', edgecolor='black')
    plt.suptitle('Feature Distributions', fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_feature_target_relationship(features, targets, key_features):
    """
    Plot scatter plots for key features against the target variable.
    
    Args:
        features (pd.DataFrame): Feature dataset.
        targets (pd.Series): Target variable.
        key_features (list): List of feature names to plot.
    """
    for feature in key_features:
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=features[feature], y=targets, color="purple", alpha=0.6)
        plt.title(f'{feature} vs. Y1')
        plt.xlabel(feature)
        plt.ylabel('Y1')
        plt.show()


def plot_temporal_analysis(features, targets, feature_name, time_column='time'):
    """
    Plot temporal trends for a given feature and the target variable.
    
    Args:
        features (pd.DataFrame): Feature dataset.
        targets (pd.Series): Target variable.
        feature_name (str): Feature name for the temporal plot.
        time_column (str): Name of the time column.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(features[time_column], features[feature_name], label=feature_name, color='orange', alpha=0.8)
    plt.title(f'{feature_name} Over Time')
    plt.xlabel('Time')
    plt.ylabel(feature_name)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(features[time_column], targets, label='Y1', color='blue', alpha=0.8)
    plt.title('Target (Y1) Over Time')
    plt.xlabel('Time')
    plt.ylabel('Y1')
    plt.legend()
    plt.show()


def plot_class_separability(features, targets, key_features, binary_threshold):
    """
    Plot class separability using pair plots.
    
    Args:
        features (pd.DataFrame): Feature dataset.
        targets (pd.Series): Target variable.
        key_features (list): List of feature names to plot.
        binary_threshold (int or float): Threshold for binary classification.
    """
    binary_target = (targets >= binary_threshold).astype(int)
    sns.pairplot(features.assign(Target=binary_target), vars=key_features, hue="Target", palette="coolwarm", diag_kind="kde")
    plt.suptitle('Feature Relationships by Target', y=1.02, fontsize=16)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels=None, normalize=False, title="Confusion Matrix", cmap="Blues"):
    """
    Plots a confusion matrix using seaborn for better visuals.
    
    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        labels (list, optional): List of class labels. Defaults to None.
        normalize (bool): Whether to normalize the matrix by row sums. Defaults to False.
        title (str): Title for the plot. Defaults to "Confusion Matrix".
        cmap (str): Colormap for the heatmap. Defaults to "Blues".
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create a DataFrame for seaborn heatmap
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap, cbar=True, square=True)
    plt.title(title, fontsize=16)
    plt.xlabel("Predicted Labels", fontsize=12)
    plt.ylabel("True Labels", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10, rotation=0)
    plt.tight_layout()
    plt.show()
