import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.datasets import make_classification, make_regression, load_iris, load_wine, load_breast_cancer
from typing import Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """
    A utility class for data preprocessing and preparation.
    """
    
    def __init__(self):
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
    
    def load_sample_dataset(self, dataset_name: str = 'iris') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load a sample dataset for testing.
        
        Args:
            dataset_name: Name of dataset ('iris', 'wine', 'breast_cancer', 'synthetic_classification', 'synthetic_regression')
        
        Returns:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
        """
        if dataset_name == 'iris':
            data = load_iris()
            return data.data, data.target, list(data.feature_names)
        
        elif dataset_name == 'wine':
            data = load_wine()
            return data.data, data.target, list(data.feature_names)
        
        elif dataset_name == 'breast_cancer':
            data = load_breast_cancer()
            return data.data, data.target, list(data.feature_names)
        
        elif dataset_name == 'synthetic_classification':
            X, y = make_classification(
                n_samples=1000, 
                n_features=20, 
                n_informative=15, 
                n_redundant=5, 
                n_classes=3, 
                random_state=42
            )
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            return X, y, feature_names
        
        elif dataset_name == 'synthetic_regression':
            X, y = make_regression(
                n_samples=1000, 
                n_features=10, 
                noise=0.1, 
                random_state=42
            )
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            return X, y, feature_names
        
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def preprocess_data(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                       normalize: bool = True, encode_labels: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess the data by normalizing features and encoding labels.
        
        Args:
            X: Feature matrix
            y: Target vector (optional)
            normalize: Whether to normalize features
            encode_labels: Whether to encode categorical labels
        
        Returns:
            Preprocessed X and y
        """
        # Normalize features
        if normalize:
            if self.scaler is None:
                self.scaler = StandardScaler()
                X_processed = self.scaler.fit_transform(X)
            else:
                X_processed = self.scaler.transform(X)
        else:
            X_processed = X.copy()
        
        # Encode labels
        y_processed = y
        if y is not None and encode_labels:
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
                y_processed = self.label_encoder.fit_transform(y)
            else:
                y_processed = self.label_encoder.transform(y)
        
        return X_processed, y_processed
    
    def split_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
                   val_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, ...]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            random_state: Random seed
        
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: train and val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_polynomial_features(self, X: np.ndarray, degree: int = 2) -> np.ndarray:
        """
        Create polynomial features.
        
        Args:
            X: Feature matrix
            degree: Polynomial degree
        
        Returns:
            Extended feature matrix with polynomial terms
        """
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        return poly.fit_transform(X)
    
    def handle_missing_values(self, X: np.ndarray, strategy: str = 'mean') -> np.ndarray:
        """
        Handle missing values in the data.
        
        Args:
            X: Feature matrix with potential missing values
            strategy: Imputation strategy ('mean', 'median', 'most_frequent')
        
        Returns:
            Feature matrix with imputed values
        """
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy=strategy)
        return imputer.fit_transform(X)
    
    def visualize_data(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> None:
        """
        Create visualizations of the data.
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: Names of features
        """
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        
        # Create DataFrame for easier plotting
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        # Set up the plotting area
        n_features = min(X.shape[1], 6)  # Limit to 6 features for readability
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        # Plot distributions of first few features
        for i in range(n_features):
            if i < len(feature_names):
                sns.histplot(data=df, x=feature_names[i], hue='target', ax=axes[i], alpha=0.7)
                axes[i].set_title(f'Distribution of {feature_names[i]}')
                axes[i].legend()
        
        plt.tight_layout()
        plt.show()
        
        # Correlation matrix
        if X.shape[1] <= 20:  # Only for manageable number of features
            plt.figure(figsize=(12, 10))
            correlation_matrix = df.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, linewidths=0.5)
            plt.title('Feature Correlation Matrix')
            plt.show()
        
        # Pairplot for small datasets
        if X.shape[1] <= 6 and len(np.unique(y)) <= 5:
            plt.figure(figsize=(15, 12))
            sns.pairplot(df, hue='target', diag_kind='hist', markers='o', alpha=0.7)
            plt.suptitle('Pairwise Feature Relationships', y=1.02)
            plt.show()


def generate_synthetic_dataset(n_samples: int = 1000, n_features: int = 10, 
                             task_type: str = 'classification', noise: float = 0.1, 
                             random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic dataset for testing.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        task_type: 'classification' or 'regression'
        noise: Amount of noise to add
        random_state: Random seed
    
    Returns:
        X: Feature matrix
        y: Target vector
    """
    np.random.seed(random_state)
    
    if task_type == 'classification':
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=max(2, n_features // 2),
            n_redundant=min(2, n_features // 4),
            n_classes=3,
            flip_y=noise,
            random_state=random_state
        )
    elif task_type == 'regression':
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=noise * 10,
            random_state=random_state
        )
    else:
        raise ValueError("task_type must be 'classification' or 'regression'")
    
    return X, y


def create_spiral_dataset(n_samples: int = 300) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a spiral dataset for testing non-linear models.
    
    Args:
        n_samples: Number of samples per class
    
    Returns:
        X: 2D feature matrix
        y: Binary target vector
    """
    np.random.seed(42)
    
    N = n_samples // 2  # number of points per class
    D = 2  # dimensionality
    X = np.zeros((n_samples, D))
    y = np.zeros(n_samples, dtype='uint8')
    
    for j in range(2):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N)  # radius
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    
    return X, y


def plot_decision_boundary(model, X: np.ndarray, y: np.ndarray, title: str = "Decision Boundary") -> None:
    """
    Plot the decision boundary for a 2D dataset.
    
    Args:
        model: Trained model with predict method
        X: 2D feature matrix
        y: Target vector
        title: Plot title
    """
    if X.shape[1] != 2:
        print("Decision boundary plotting only works for 2D data")
        return
    
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # Generate a grid of points
    h = 0.02  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    # Make predictions on the grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    try:
        if hasattr(model, 'predict_proba'):
            Z = model.predict_proba(grid_points)
            if Z.shape[1] > 1:
                Z = Z[:, 1]  # Use probability of positive class
        else:
            Z = model.predict(grid_points)
    except:
        # Fallback for models that don't support the grid format
        Z = np.array([model.predict(point.reshape(1, -1))[0] for point in grid_points])
    
    Z = Z.reshape(xx.shape)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    
    # Plot the data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
    plt.colorbar(scatter)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.show()


def evaluate_model_performance(y_true: np.ndarray, y_pred: np.ndarray, 
                             task_type: str = 'classification') -> dict:
    """
    Evaluate model performance with various metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        task_type: 'classification' or 'regression'
    
    Returns:
        Dictionary of performance metrics
    """
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                                confusion_matrix, mean_squared_error, mean_absolute_error, r2_score)
    
    metrics = {}
    
    if task_type == 'classification':
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Handle multi-class vs binary classification
        if len(np.unique(y_true)) > 2:
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
            metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')
        else:
            metrics['precision'] = precision_score(y_true, y_pred)
            metrics['recall'] = recall_score(y_true, y_pred)
            metrics['f1_score'] = f1_score(y_true, y_pred)
        
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
    elif task_type == 'regression':
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2_score'] = r2_score(y_true, y_pred)
    
    return metrics


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         class_names: Optional[List[str]] = None) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()