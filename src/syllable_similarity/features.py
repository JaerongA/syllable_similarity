"""
Feature engineering functions for syllable similarity analysis.
"""

import numpy as np
import pandas as pd
from scipy import signal

from .config import PROCESSED_DATA_DIR


def extract_spectral_features(audio_data, sample_rate):
    """
    Extract spectral features from audio data.

    Parameters:
    -----------
    audio_data : np.ndarray
        Audio waveform data
    sample_rate : int
        Sampling rate of the audio

    Returns:
    --------
    dict
        Dictionary containing extracted spectral features
    """
    # Calculate power spectral density
    frequencies, psd = signal.welch(audio_data, fs=sample_rate)

    # Extract spectral features
    features = {
        "frequencies": frequencies,
        "psd": psd,
        "spectral_centroid": np.sum(frequencies * psd) / np.sum(psd),
        "spectral_bandwidth": np.sqrt(
            np.sum(((frequencies - np.mean(frequencies)) ** 2) * psd) / np.sum(psd)
        ),
        "spectral_rolloff": np.percentile(frequencies, 85),
        "spectral_flatness": np.exp(np.mean(np.log(psd + 1e-10))) / np.mean(psd),
    }

    return features


def calculate_similarity_matrix(features_list):
    """
    Calculate similarity matrix between multiple feature sets.

    Parameters:
    -----------
    features_list : list
        List of feature dictionaries

    Returns:
    --------
    np.ndarray
        Similarity matrix
    """
    n_features = len(features_list)
    similarity_matrix = np.zeros((n_features, n_features))

    for i in range(n_features):
        for j in range(n_features):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                # Calculate cosine similarity between PSD vectors
                psd_i = features_list[i]["psd"]
                psd_j = features_list[j]["psd"]

                # Normalize PSD vectors
                psd_i_norm = psd_i / (np.linalg.norm(psd_i) + 1e-10)
                psd_j_norm = psd_j / (np.linalg.norm(psd_j) + 1e-10)

                # Cosine similarity
                similarity = np.dot(psd_i_norm, psd_j_norm)
                similarity_matrix[i, j] = similarity

    return similarity_matrix


def create_feature_dataframe(features_list):
    """
    Create a pandas DataFrame from extracted features.

    Parameters:
    -----------
    features_list : list
        List of feature dictionaries

    Returns:
    --------
    pd.DataFrame
        DataFrame containing extracted features
    """
    feature_data = []

    for i, features in enumerate(features_list):
        row = {
            "sample_id": i,
            "spectral_centroid": features["spectral_centroid"],
            "spectral_bandwidth": features["spectral_bandwidth"],
            "spectral_rolloff": features["spectral_rolloff"],
            "spectral_flatness": features["spectral_flatness"],
        }
        feature_data.append(row)

    return pd.DataFrame(feature_data)
