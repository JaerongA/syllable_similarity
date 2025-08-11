"""
Model training functions for syllable similarity analysis.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from ..config import MODELS_DIR


def train_similarity_classifier(features, labels, test_size=0.2, random_state=42):
    """
    Train a classifier for syllable similarity classification.

    Parameters:
    -----------
    features : np.ndarray or pd.DataFrame
        Feature matrix for training
    labels : np.ndarray or pd.Series
        Target labels for training
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    tuple
        Trained model and test features/labels
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state
    )

    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)

    # Make predictions on test set
    y_pred = model.predict(X_test)

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model, (X_test, y_test, y_pred)


def save_model(model, model_name="similarity_classifier", model_dir=None):
    """
    Save a trained model to disk.

    Parameters:
    -----------
    model : sklearn estimator
        Trained model to save
    model_name : str
        Name for the saved model file
    model_dir : str or Path, optional
        Directory to save the model. If None, uses default models directory.
    """
    if model_dir is None:
        model_dir = MODELS_DIR

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{model_name}.joblib"
    joblib.dump(model, model_path)

    print(f"Model saved to {model_path}")


def load_model(model_name="similarity_classifier", model_dir=None):
    """
    Load a trained model from disk.

    Parameters:
    -----------
    model_name : str
        Name of the model file to load
    model_dir : str or Path, optional
        Directory containing the model. If None, uses default models directory.

    Returns:
    --------
    sklearn estimator
        Loaded trained model
    """
    if model_dir is None:
        model_dir = MODELS_DIR

    model_dir = Path(model_dir)
    model_path = model_dir / f"{model_name}.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    return joblib.load(model_path)
