"""
Input/output utilities for GaN-DefectML.

This module provides reusable helpers for:
- CSV and JSON persistence,
- Pymatgen structure serialization,
- structure-library persistence,
- feature and target tables,
- PyTorch Geometric graph datasets,
- PyTorch model checkpoints,
- prediction and metric outputs,
- and filesystem validation.

Model architectures are defined separately in the top-level
``models`` package.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import torch

from pymatgen.core import Structure


# =====================================================================
# Generic path helpers
# =====================================================================

def ensure_parent_directory(
    file_path,
) -> Path:
    """
    Ensure that the parent directory of a file exists.
    """

    file_path = Path(
        file_path
    )

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return file_path


def ensure_directory(
    directory_path,
) -> Path:
    """
    Create a directory if it does not already exist.
    """

    directory_path = Path(
        directory_path
    )

    directory_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return directory_path


def require_file(
    file_path,
) -> Path:
    """
    Validate that a required file exists.
    """

    file_path = Path(
        file_path
    )

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required file was not found: "
            f"{file_path}"
        )

    if not file_path.is_file():

        raise ValueError(
            f"Expected a file but found: "
            f"{file_path}"
        )

    return file_path


# =====================================================================
# JSON
# =====================================================================

def _convert_to_json_serializable(
    value: Any,
) -> Any:
    """
    Convert common scientific Python objects into JSON-safe objects.
    """

    if isinstance(
        value,
        Path,
    ):

        return str(
            value
        )

    if isinstance(
        value,
        np.ndarray,
    ):

        return value.tolist()

    if isinstance(
        value,
        np.integer,
    ):

        return int(
            value
        )

    if isinstance(
        value,
        np.floating,
    ):

        return float(
            value
        )

    if isinstance(
        value,
        np.bool_,
    ):

        return bool(
            value
        )

    if isinstance(
        value,
        torch.Tensor,
    ):

        return (
            value
            .detach()
            .cpu()
            .tolist()
        )

    if isinstance(
        value,
        dict,
    ):

        return {
            str(key):
                _convert_to_json_serializable(
                    item
                )
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (list, tuple, set),
    ):

        return [
            _convert_to_json_serializable(
                item
            )
            for item in value
        ]

    return value


def save_json(
    data: Dict[str, Any],
    output_path,
    indent: int = 4,
) -> Path:
    """
    Save a dictionary as JSON.
    """

    output_path = (
        ensure_parent_directory(
            output_path
        )
    )

    serializable_data = (
        _convert_to_json_serializable(
            data
        )
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            serializable_data,
            file,
            indent=indent,
        )

    return output_path


def load_json(
    input_path,
) -> dict:
    """
    Load a JSON file.
    """

    input_path = require_file(
        input_path
    )

    with open(
        input_path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


# =====================================================================
# DataFrames / CSV
# =====================================================================

def save_dataframe(
    dataframe: pd.DataFrame,
    output_path,
    index: bool = False,
) -> Path:
    """
    Save a pandas DataFrame as CSV.
    """

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):

        raise TypeError(
            "Expected a pandas DataFrame."
        )

    output_path = (
        ensure_parent_directory(
            output_path
        )
    )

    dataframe.to_csv(
        output_path,
        index=index,
    )

    return output_path


def load_dataframe(
    input_path,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """
    Load a CSV file as a pandas DataFrame.
    """

    input_path = require_file(
        input_path
    )

    return pd.read_csv(
        input_path,
        **read_csv_kwargs,
    )


# =====================================================================
# Numerical arrays
# =====================================================================

def save_numpy_array(
    array,
    output_path,
) -> Path:
    """
    Save a numerical array in NumPy binary format.
    """

    output_path = (
        ensure_parent_directory(
            output_path
        )
    )

    np.save(
        output_path,
        np.asarray(
            array
        ),
    )

    return output_path


def load_numpy_array(
    input_path,
) -> np.ndarray:
    """
    Load a NumPy binary array.
    """

    input_path = require_file(
        input_path
    )

    return np.load(
        input_path,
        allow_pickle=False,
    )


# =====================================================================
# Individual crystal structures
# =====================================================================

def save_structure(
    structure: Structure,
    output_path,
) -> Path:
    """
    Save a Pymatgen Structure.

    The file format is inferred from the extension. Recommended
    formats are CIF and POSCAR.
    """

    if not isinstance(
        structure,
        Structure,
    ):

        raise TypeError(
            "structure must be a "
            "pymatgen.core.Structure object."
        )

    output_path = (
        ensure_parent_directory(
            output_path
        )
    )

    structure.to(
        filename=str(
            output_path
        )
    )

    return output_path


def load_structure(
    input_path,
) -> Structure:
    """
    Load a crystal structure using Pymatgen.
    """

    input_path = require_file(
        input_path
    )

    return Structure.from_file(
        str(
            input_path
        )
    )


# =====================================================================
# Structure library
# =====================================================================

def save_structure_library(
    structure_library: Dict[str, Structure],
    output_directory,
    file_format: str = "cif",
) -> pd.DataFrame:
    """
    Save the complete defect structure library.

    Parameters
    ----------
    structure_library
        Mapping from configuration ID to Pymatgen Structure.

    output_directory
        Directory in which structures are written.

    file_format
        Structure format, normally ``cif``.

    Returns
    -------
    pandas.DataFrame
        Manifest containing configuration IDs and file paths.
    """

    output_directory = (
        ensure_directory(
            output_directory
        )
    )

    file_format = (
        file_format
        .lower()
        .lstrip(".")
    )

    records = []

    for configuration_id, structure in (
        structure_library.items()
    ):

        file_path = (
            output_directory
            / (
                f"{configuration_id}."
                f"{file_format}"
            )
        )

        save_structure(
            structure=structure,
            output_path=file_path,
        )

        records.append(
            {
                "configuration_id":
                    configuration_id,

                "structure_file":
                    str(
                        file_path
                    ),

                "number_of_sites":
                    int(
                        len(
                            structure
                        )
                    ),

                "formula":
                    structure
                    .composition
                    .reduced_formula,
            }
        )

    manifest_df = pd.DataFrame(
        records
    )

    save_dataframe(
        manifest_df,
        output_directory
        / "structure_manifest.csv",
    )

    return manifest_df


def load_structure_library(
    manifest_path,
) -> Dict[str, Structure]:
    """
    Reconstruct a structure library from its manifest.
    """

    manifest_path = require_file(
        manifest_path
    )

    manifest_df = pd.read_csv(
        manifest_path
    )

    required_columns = {
        "configuration_id",
        "structure_file",
    }

    missing_columns = (
        required_columns
        - set(
            manifest_df.columns
        )
    )

    if missing_columns:

        raise ValueError(
            "Structure manifest is missing "
            f"columns: {sorted(missing_columns)}"
        )

    structure_library = {}

    for _, row in (
        manifest_df.iterrows()
    ):

        structure_path = Path(
            row[
                "structure_file"
            ]
        )

        # Support manifests moved together with their structure folder.
        if not structure_path.exists():

            alternative_path = (
                manifest_path.parent
                / structure_path.name
            )

            structure_path = (
                alternative_path
            )

        structure_library[
            row[
                "configuration_id"
            ]
        ] = load_structure(
            structure_path
        )

    return structure_library


# =====================================================================
# Feature matrices
# =====================================================================

def save_feature_matrix(
    feature_dataframe: pd.DataFrame,
    output_path,
    configuration_column: str = "configuration_id",
) -> Path:
    """
    Save a physics-informed feature matrix.
    """

    if (
        configuration_column
        not in feature_dataframe.columns
    ):

        raise ValueError(
            f"Feature matrix must contain "
            f"'{configuration_column}'."
        )

    if feature_dataframe[
        configuration_column
    ].duplicated().any():

        raise ValueError(
            "Feature matrix contains duplicate "
            "configuration IDs."
        )

    return save_dataframe(
        feature_dataframe,
        output_path,
        index=False,
    )


def load_feature_matrix(
    input_path,
    configuration_column: str = "configuration_id",
) -> pd.DataFrame:
    """
    Load and minimally validate a feature matrix.
    """

    feature_dataframe = (
        load_dataframe(
            input_path
        )
    )

    if (
        configuration_column
        not in feature_dataframe.columns
    ):

        raise ValueError(
            f"Feature matrix does not contain "
            f"'{configuration_column}'."
        )

    return feature_dataframe


# =====================================================================
# Target tables
# =====================================================================

def save_target_table(
    target_dataframe: pd.DataFrame,
    output_path,
    configuration_column: str = "configuration_id",
) -> Path:
    """
    Save defect-property target data.
    """

    if (
        configuration_column
        not in target_dataframe.columns
    ):

        raise ValueError(
            f"Target table must contain "
            f"'{configuration_column}'."
        )

    if target_dataframe[
        configuration_column
    ].duplicated().any():

        raise ValueError(
            "Target table contains duplicate "
            "configuration IDs."
        )

    return save_dataframe(
        target_dataframe,
        output_path,
        index=False,
    )


def load_target_table(
    input_path,
    configuration_column: str = "configuration_id",
) -> pd.DataFrame:
    """
    Load the defect-property target table.
    """

    target_dataframe = (
        load_dataframe(
            input_path
        )
    )

    if (
        configuration_column
        not in target_dataframe.columns
    ):

        raise ValueError(
            f"Target table does not contain "
            f"'{configuration_column}'."
        )

    return target_dataframe


# =====================================================================
# Graph datasets
# =====================================================================

def save_graph_dataset(
    graphs: Iterable,
    output_path,
) -> Path:
    """
    Save a PyTorch Geometric graph dataset.

    The dataset is stored using torch.save.
    """

    graphs = list(
        graphs
    )

    if len(
        graphs
    ) == 0:

        raise ValueError(
            "Cannot save an empty graph dataset."
        )

    output_path = (
        ensure_parent_directory(
            output_path
        )
    )

    torch.save(
        graphs,
        output_path,
    )

    return output_path


def load_graph_dataset(
    input_path,
    map_location: str | torch.device = "cpu",
):
    """
    Load a serialized PyG graph dataset.

    Notes
    -----
    Only load graph files produced by this project or another trusted
    source because PyTorch serialization is not intended for
    untrusted files.
    """

    input_path = require_file(
        input_path
    )

    graphs = torch.load(
        input_path,
        map_location=
            map_location,
        weights_only=False,
    )

    if not isinstance(
        graphs,
        (list, tuple),
    ):

        raise TypeError(
            "Serialized graph dataset must "
            "contain a list or tuple."
        )

    return list(
        graphs
    )


# =====================================================================
# PyTorch checkpoints
# =====================================================================

def save_torch_checkpoint(
    model,
    output_path,
    optimizer=None,
    epoch: Optional[int] = None,
    metrics: Optional[
        Dict[str, Any]
    ] = None,
    metadata: Optional[
        Dict[str, Any]
    ] = None,
) -> Path:
    """
    Save a PyTorch model state checkpoint.

    The architecture itself is not serialized. The corresponding model
    class should be instantiated from the top-level ``models`` package
    before loading the state dictionary.
    """

    output_path = (
        ensure_parent_directory(
            output_path
        )
    )

    checkpoint = {
        "model_state_dict":
            model.state_dict(),

        "epoch":
            epoch,

        "metrics":
            metrics or {},

        "metadata":
            metadata or {},
    }

    if optimizer is not None:

        checkpoint[
            "optimizer_state_dict"
        ] = optimizer.state_dict()

    torch.save(
        checkpoint,
        output_path,
    )

    return output_path


def load_torch_checkpoint(
    input_path,
    model,
    optimizer=None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict:
    """
    Load a PyTorch state checkpoint into an instantiated model.
    """

    input_path = require_file(
        input_path
    )

    checkpoint = torch.load(
        input_path,
        map_location=
            map_location,
        weights_only=False,
    )

    if (
        "model_state_dict"
        not in checkpoint
    ):

        raise KeyError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=strict,
    )

    if (
        optimizer is not None
        and
        "optimizer_state_dict"
        in checkpoint
    ):

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

    return checkpoint


# =====================================================================
# Result persistence
# =====================================================================

def save_metrics(
    metrics: Dict[str, Any],
    output_path,
) -> Path:
    """
    Save model evaluation metrics as JSON.
    """

    return save_json(
        metrics,
        output_path,
    )


def save_predictions(
    predictions,
    output_path,
) -> Path:
    """
    Save prediction results.

    Accepts either a DataFrame or an iterable of dictionaries.
    """

    if isinstance(
        predictions,
        pd.DataFrame,
    ):

        prediction_df = (
            predictions.copy()
        )

    else:

        prediction_df = (
            pd.DataFrame(
                predictions
            )
        )

    return save_dataframe(
        prediction_df,
        output_path,
        index=False,
    )


# =====================================================================
# Manifest utilities
# =====================================================================

def build_file_manifest(
    directory,
    recursive: bool = True,
) -> pd.DataFrame:
    """
    Build a simple manifest of files in a project directory.
    """

    directory = Path(
        directory
    )

    if not directory.exists():

        raise FileNotFoundError(
            f"Directory not found: "
            f"{directory}"
        )

    iterator = (
        directory.rglob("*")
        if recursive
        else directory.glob("*")
    )

    records = []

    for file_path in iterator:

        if not file_path.is_file():
            continue

        records.append(
            {
                "file_name":
                    file_path.name,

                "relative_path":
                    str(
                        file_path.relative_to(
                            directory
                        )
                    ),

                "suffix":
                    file_path.suffix,

                "size_bytes":
                    int(
                        file_path.stat().st_size
                    ),
            }
        )

    return (
        pd.DataFrame(
            records
        )
        .sort_values(
            "relative_path"
        )
        .reset_index(
            drop=True
        )
        if records
        else pd.DataFrame(
            columns=[
                "file_name",
                "relative_path",
                "suffix",
                "size_bytes",
            ]
        )
    )
