"""
Visualization functions for syllable similarity analysis.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import FIGURES_DIR


def setup_plotting_style():
    """Set up consistent plotting style for all figures."""
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")

    # Set figure parameters
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12


def plot_spectrogram(audio_data, sample_rate, title="Spectrogram", save_path=None):
    """
    Plot spectrogram of audio data.

    Parameters:
    -----------
    audio_data : np.ndarray
        Audio waveform data
    sample_rate : int
        Sampling rate of the audio
    title : str
        Title for the plot
    save_path : str or Path, optional
        Path to save the figure
    """
    setup_plotting_style()

    plt.figure(figsize=(12, 6))
    plt.specgram(audio_data, Fs=sample_rate, cmap="viridis")
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Power/Frequency (dB/Hz)")

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_similarity_heatmap(
    similarity_matrix, labels=None, title="Similarity Matrix", save_path=None
):
    """
    Plot similarity matrix as a heatmap.

    Parameters:
    -----------
    similarity_matrix : np.ndarray
        Similarity matrix to plot
    labels : list, optional
        Labels for the axes
    title : str
        Title for the plot
    save_path : str or Path, optional
        Path to save the figure
    """
    setup_plotting_style()

    plt.figure(figsize=(10, 8))

    if labels is None:
        labels = [f"Sample {i}" for i in range(len(similarity_matrix))]

    sns.heatmap(
        similarity_matrix,
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        cmap="RdYlBu_r",
        vmin=0,
        vmax=1,
        square=True,
    )

    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Samples")

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_psd_comparison(
    frequencies_list,
    psd_list,
    labels=None,
    title="Power Spectral Density Comparison",
    save_path=None,
):
    """
    Plot comparison of power spectral densities.

    Parameters:
    -----------
    frequencies_list : list
        List of frequency arrays
    psd_list : list
        List of PSD arrays
    labels : list, optional
        Labels for each PSD
    title : str
        Title for the plot
    save_path : str or Path, optional
        Path to save the figure
    """
    setup_plotting_style()

    plt.figure(figsize=(12, 8))

    if labels is None:
        labels = [f"Sample {i}" for i in range(len(psd_list))]

    for i, (freq, psd) in enumerate(zip(frequencies_list, psd_list)):
        plt.plot(freq, psd, label=labels[i], linewidth=2)

    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_feature_distributions(feature_df, save_path=None):
    """
    Plot distributions of extracted features.

    Parameters:
    -----------
    feature_df : pd.DataFrame
        DataFrame containing extracted features
    save_path : str or Path, optional
        Path to save the figure
    """
    setup_plotting_style()

    # Exclude non-numeric columns
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != "sample_id"]

    n_features = len(numeric_cols)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()

    for i, feature in enumerate(numeric_cols):
        if i < len(axes):
            axes[i].hist(feature_df[feature], bins=20, alpha=0.7, edgecolor="black")
            axes[i].set_title(f"{feature} Distribution")
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel("Frequency")
            axes[i].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
