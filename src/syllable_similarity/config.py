"""
Configuration management for the syllable similarity project.
"""

import json
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Reports directory
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Models directory
MODELS_DIR = PROJECT_ROOT / "models"

# Default configuration
DEFAULT_CONFIG = {
    "project_dir": "/Box/Data/BMI",
    "birdID": ["k77r57"],
    "sessions": ["pre-control1", "pre-control2", "BMI"],
    "condition": ["baseline", "BMI"],
}


def load_config(config_path=None):
    """Load configuration from JSON file."""
    if config_path is None:
        config_path = PROJECT_ROOT / "src" / "syllable_similarity" / "config.json"

    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)
    else:
        return DEFAULT_CONFIG.copy()


def save_config(config, config_path=None):
    """Save configuration to JSON file."""
    if config_path is None:
        config_path = PROJECT_ROOT / "src" / "syllable_similarity" / "config.json"

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


# Load default configuration
config = load_config()
