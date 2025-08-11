"""
Model prediction functions for syllable similarity analysis.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .train import load_model


def predict_similarity(model, features):
    """
    Make predictions using a trained similarity classifier.

    Parameters:
    -----------
    model : sklearn estimator
        Trained model for making predictions
    features : np.ndarray or pd.DataFrame
        Feature matrix for prediction

    Returns:
    --------
    np.ndarray
        Predicted similarity classes
    """
    predictions = model.predict(features)
    return predictions


def predict_similarity_proba(model, features):
    """
    Make probability predictions using a trained similarity classifier.

    Parameters:
    -----------
    model : sklearn estimator
        Trained model for making predictions
    features : np.ndarray or pd.DataFrame
        Feature matrix for prediction

    Returns:
    --------
    np.ndarray
        Predicted probability for each class
    """
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        return probabilities
    else:
        raise AttributeError("Model does not support probability predictions")


def batch_predict_similarity(
    model_name="similarity_classifier", features_file=None, output_file=None
):
    """
    Perform batch predictions on a dataset.

    Parameters:
    -----------
    model_name : str
        Name of the model to load
    features_file : str or Path, optional
        Path to CSV file containing features. If None, uses default processed data.
    output_file : str or Path, optional
        Path to save prediction results. If None, saves to reports directory.
    """
    # Load the trained model
    model = load_model(model_name)

    # Load features
    if features_file is None:
        from ..config import PROCESSED_DATA_DIR

        features_file = PROCESSED_DATA_DIR / "processed_similarity.csv"

    features_df = pd.read_csv(features_file)

    # Prepare features (exclude non-feature columns)
    feature_cols = [
        col for col in features_df.columns if col not in ["sample_id", "label"]
    ]
    X = features_df[feature_cols].values

    # Make predictions
    predictions = predict_similarity(model, X)

    # Create results DataFrame
    results_df = features_df.copy()
    results_df["predicted_class"] = predictions

    # Add probabilities if available
    try:
        probabilities = predict_similarity_proba(model, X)
        for i, class_name in enumerate(model.classes_):
            results_df[f"prob_class_{class_name}"] = probabilities[:, i]
    except AttributeError:
        pass

    # Save results
    if output_file is None:
        from ..config import REPORTS_DIR

        output_file = REPORTS_DIR / "prediction_results.csv"

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_file, index=False)

    print(f"Prediction results saved to {output_file}")

    return results_df


def evaluate_predictions(y_true, y_pred, save_report=True, output_dir=None):
    """
    Evaluate prediction performance and generate a report.

    Parameters:
    -----------
    y_true : np.ndarray or pd.Series
        True labels
    y_pred : np.ndarray or pd.Series
        Predicted labels
    save_report : bool
        Whether to save the evaluation report
    output_dir : str or Path, optional
        Directory to save the report. If None, uses default reports directory.

    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Compile results
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
    }

    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")

    # Save report if requested
    if save_report:
        if output_dir is None:
            from ..config import REPORTS_DIR

            output_dir = REPORTS_DIR

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics to CSV
        metrics_df = pd.DataFrame(
            [results], columns=["accuracy", "precision", "recall", "f1_score"]
        )
        metrics_df.to_csv(output_dir / "evaluation_metrics.csv", index=False)

        # Save confusion matrix
        cm_df = pd.DataFrame(cm)
        cm_df.to_csv(output_dir / "confusion_matrix.csv", index=False)

        print(f"Evaluation report saved to {output_dir}")

    return results
