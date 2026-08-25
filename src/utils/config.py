"""
Central configuration utilities for GaN-DefectML.

This module defines:
- project paths,
- reproducibility settings,
- graph-construction parameters,
- supervised-learning thresholds,
- GNN hyperparameters,
- explainability settings,
- and configuration serialization helpers.

Individual modules may override these values when required, but these
defaults represent the validated project configuration.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# =====================================================================
# Project paths
# =====================================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
)

RAW_DATA_DIR = (
    DATA_DIR
    / "raw"
)

PROCESSED_DATA_DIR = (
    DATA_DIR
    / "processed"
)

TARGET_DATA_DIR = (
    DATA_DIR
    / "targets"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

FIGURES_DIR = (
    RESULTS_DIR
    / "figures"
)

METRICS_DIR = (
    RESULTS_DIR
    / "metrics"
)

PREDICTIONS_DIR = (
    RESULTS_DIR
    / "predictions"
)

CONFIG_DIR = (
    PROJECT_ROOT
    / "configs"
)


# =====================================================================
# Configuration classes
# =====================================================================

@dataclass
class GraphConfig:
    """
    Crystal graph construction configuration.
    """

    cutoff_radius_A: float = 3.0
    number_of_rbf: int = 16

    node_feature_dimension: int = 16
    edge_feature_dimension: int = 17
    global_feature_dimension: int = 28


@dataclass
class LearningConfig:
    """
    Supervised-learning configuration.
    """

    random_seed: int = 42

    primary_regression_target: str = (
        "formation_energy_eV"
    )

    secondary_regression_targets: tuple = (
        "relaxed_band_gap_eV",
        "band_gap_change_eV",
        "transition_level_eV",
        "carrier_concentration_cm3",
    )

    classification_targets: tuple = (
        "donor_acceptor_class",
        "carrier_type",
    )

    minimum_regression_samples: int = 30

    minimum_classification_samples: int = 30

    minimum_samples_per_class: int = 5

    classical_cv_folds: int = 5

    gnn_cv_folds: int = 5

    gnn_batch_size: int = 8

    gnn_hidden_dim: int = 64

    gnn_dropout: float = 0.2

    gnn_learning_rate: float = 1e-3

    gnn_weight_decay: float = 1e-5

    gnn_max_epochs: int = 300

    gnn_patience: int = 30

    primary_graph_cutoff_A: float = 3.0


@dataclass
class XAIConfig:
    """
    Explainable-AI configuration.
    """

    permutation_repeats: int = 30

    random_seed: int = 42

    top_features_to_display: int = 15

    node_mask_baseline: str = "zero"

    edge_mask_baseline: str = "zero"

    graph_attribute_mask_baseline: str = (
        "zero"
    )

    diagnostic_mode: bool = True


@dataclass
class ProjectConfig:
    """
    Complete GaN-DefectML configuration.
    """

    graph: GraphConfig
    learning: LearningConfig
    xai: XAIConfig


# =====================================================================
# Default configuration
# =====================================================================

def get_default_config() -> ProjectConfig:
    """
    Return the default project configuration.
    """

    return ProjectConfig(
        graph=GraphConfig(),
        learning=LearningConfig(),
        xai=XAIConfig(),
    )


# =====================================================================
# Directory management
# =====================================================================

def create_project_directories() -> None:
    """
    Create expected project output/data directories if absent.
    """

    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        TARGET_DATA_DIR,
        MODEL_DIR,
        RESULTS_DIR,
        FIGURES_DIR,
        METRICS_DIR,
        PREDICTIONS_DIR,
        CONFIG_DIR,
    ]

    for directory in directories:

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# =====================================================================
# Reproducibility
# =====================================================================

def set_global_seed(
    seed: int = 42,
) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch.
    """

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed(
            seed
        )

        torch.cuda.manual_seed_all(
            seed
        )


def configure_deterministic_torch(
    enabled: bool = True,
) -> None:
    """
    Configure deterministic PyTorch behavior where possible.

    Notes
    -----
    Some PyTorch/PyG operations may not have deterministic
    implementations on every device.
    """

    torch.backends.cudnn.deterministic = (
        enabled
    )

    torch.backends.cudnn.benchmark = (
        not enabled
    )


# =====================================================================
# Device selection
# =====================================================================

def get_device(
    prefer_cuda: bool = True,
) -> torch.device:
    """
    Select the preferred PyTorch compute device.
    """

    if (
        prefer_cuda
        and torch.cuda.is_available()
    ):

        return torch.device(
            "cuda"
        )

    return torch.device(
        "cpu"
    )


def get_device_summary() -> dict:
    """
    Return basic compute-environment information.
    """

    device = get_device()

    summary = {
        "torch_version":
            torch.__version__,

        "cuda_available":
            bool(
                torch.cuda.is_available()
            ),

        "selected_device":
            str(
                device
            ),
    }

    if torch.cuda.is_available():

        summary[
            "cuda_device_name"
        ] = torch.cuda.get_device_name(
            0
        )

    else:

        summary[
            "cuda_device_name"
        ] = None

    return summary


# =====================================================================
# Configuration serialization
# =====================================================================

def config_to_dict(
    config: ProjectConfig,
) -> dict:
    """
    Convert project configuration to a serializable dictionary.
    """

    return asdict(
        config
    )


def save_config(
    config: ProjectConfig,
    output_path: Optional[
        Path
    ] = None,
) -> Path:
    """
    Save complete project configuration as JSON.
    """

    if output_path is None:

        output_path = (
            CONFIG_DIR
            / "project_config.json"
        )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            config_to_dict(
                config
            ),
            file,
            indent=4,
        )

    return output_path


def save_learning_config(
    config: LearningConfig,
    output_path: Optional[
        Path
    ] = None,
) -> Path:
    """
    Save learning configuration separately.
    """

    if output_path is None:

        output_path = (
            CONFIG_DIR
            / "learning_config.json"
        )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            asdict(
                config
            ),
            file,
            indent=4,
        )

    return output_path


def save_xai_config(
    config: XAIConfig,
    output_path: Optional[
        Path
    ] = None,
) -> Path:
    """
    Save XAI configuration separately.
    """

    if output_path is None:

        output_path = (
            CONFIG_DIR
            / "xai_config.json"
        )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            asdict(
                config
            ),
            file,
            indent=4,
        )

    return output_path


# =====================================================================
# Configuration loading
# =====================================================================

def load_json_config(
    config_path,
) -> dict:
    """
    Load a JSON configuration file.
    """

    config_path = Path(
        config_path
    )

    if not config_path.exists():

        raise FileNotFoundError(
            f"Configuration file not found: "
            f"{config_path}"
        )

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


# =====================================================================
# Initialization
# =====================================================================

def initialize_project(
    deterministic: bool = True,
) -> ProjectConfig:
    """
    Initialize directories, reproducibility, and default configuration.

    Returns
    -------
    ProjectConfig
        Active default project configuration.
    """

    config = (
        get_default_config()
    )

    create_project_directories()

    set_global_seed(
        config.learning.random_seed
    )

    configure_deterministic_torch(
        deterministic
    )

    return config
