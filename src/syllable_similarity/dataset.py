"""
Data loading and preprocessing functions for syllable similarity analysis.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROCESSED_DATA_DIR, RAW_DATA_DIR


def load_note_data(file_path=None):
    """
    Load NoteX.csv data containing note information.

    Parameters:
    -----------
    file_path : str or Path, optional
        Path to the NoteX.csv file. If None, uses default location.

    Returns:
    --------
    pd.DataFrame
        Loaded note data
    """
    if file_path is None:
        file_path = RAW_DATA_DIR / "NoteX.csv"

    if not Path(file_path).exists():
        raise FileNotFoundError(f"NoteX.csv not found at {file_path}")

    return pd.read_csv(file_path)


def load_similarity_data(file_path=None):
    """
    Load SimilarityIndex.csv data containing similarity indices.

    Parameters:
    -----------
    file_path : str or Path, optional
        Path to the SimilarityIndex.csv file. If None, uses default location.

    Returns:
    --------
    pd.DataFrame
        Loaded similarity data
    """
    if file_path is None:
        file_path = RAW_DATA_DIR / "SimilarityIndex.csv"

    if not Path(file_path).exists():
        raise FileNotFoundError(f"SimilarityIndex.csv not found at {file_path}")

    return pd.read_csv(file_path)


def preprocess_data(note_data, similarity_data):
    """
    Preprocess the loaded data for analysis.

    Parameters:
    -----------
    note_data : pd.DataFrame
        Note data from NoteX.csv
    similarity_data : pd.DataFrame
        Similarity data from SimilarityIndex.csv

    Returns:
    --------
    tuple
        Preprocessed note_data and similarity_data
    """
    # Add preprocessing steps here as needed
    # For now, just return the data as-is
    return note_data, similarity_data


def save_processed_data(note_data, similarity_data, output_dir=None):
    """
    Save preprocessed data to the processed data directory.

    Parameters:
    -----------
    note_data : pd.DataFrame
        Preprocessed note data
    similarity_data : pd.DataFrame
        Preprocessed similarity data
    output_dir : str or Path, optional
        Output directory. If None, uses default processed data directory.
    """
    if output_dir is None:
        output_dir = PROCESSED_DATA_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    note_data.to_csv(output_dir / "processed_notes.csv", index=False)
    similarity_data.to_csv(output_dir / "processed_similarity.csv", index=False)

    print(f"Processed data saved to {output_dir}")
