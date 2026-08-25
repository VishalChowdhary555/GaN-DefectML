"""
Atomic node-feature utilities for GaN-DefectML.

This module converts atomic species into physics-informed feature
vectors suitable for periodic crystal graph neural networks.

Each atom is represented using:

Continuous elemental descriptors
--------------------------------
1. Atomic number
2. Atomic mass
3. Pauling electronegativity
4. Atomic radius
5. Calculated atomic radius
6. Periodic-table row
7. Periodic-table group
8. First ionization energy
9. Electron affinity
10. Estimated valence-electron count

Categorical identity descriptors
--------------------------------
11-16. One-hot encoding for Ga, N, Mg, Si, O, and C
"""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd

from pymatgen.core import Element


GRAPH_ELEMENT_ORDER = (
    "Ga",
    "N",
    "Mg",
    "Si",
    "O",
    "C",
)


CONTINUOUS_NODE_FEATURE_NAMES = [
    "atomic_number",
    "atomic_mass",
    "electronegativity",
    "atomic_radius_A",
    "calculated_atomic_radius_A",
    "periodic_row",
    "periodic_group",
    "first_ionization_energy_eV",
    "electron_affinity_eV",
    "valence_electrons",
]


NODE_FEATURE_NAMES = (
    CONTINUOUS_NODE_FEATURE_NAMES
    + [
        f"is_{element}"
        for element in GRAPH_ELEMENT_ORDER
    ]
)


CONTINUOUS_NODE_FEATURE_COUNT = len(
    CONTINUOUS_NODE_FEATURE_NAMES
)


def safe_float(value) -> float:
    """
    Convert a scalar-like elemental property to float.

    Returns NaN when the value is unavailable or non-numeric.
    """

    if value is None:
        return np.nan

    try:
        return float(value)

    except (TypeError, ValueError):
        return np.nan


def first_ionization_energy(
    element: Element,
) -> float:
    """
    Return the first ionization energy of an element in eV.
    """

    direct_value = safe_float(
        getattr(
            element,
            "ionization_energy",
            None,
        )
    )

    if np.isfinite(
        direct_value
    ):
        return direct_value

    values = getattr(
        element,
        "ionization_energies",
        None,
    )

    try:

        if values is not None and len(values) > 0:

            return float(
                values[0]
            )

    except (
        TypeError,
        ValueError,
        IndexError,
    ):
        pass

    return np.nan


def estimate_valence_electrons(
    element: Element,
) -> float:
    """
    Estimate a conventional valence-electron descriptor from
    periodic-table group number.

    This is used only as a numerical descriptor and should not
    be interpreted as a formal oxidation-state assignment.
    """

    group = getattr(
        element,
        "group",
        None,
    )

    if group is None:
        return np.nan

    group = int(group)

    if group <= 2:
        return float(group)

    if 13 <= group <= 18:
        return float(
            group - 10
        )

    if 3 <= group <= 12:
        return float(group)

    return np.nan


def extract_node_element_properties(
    symbol: str,
) -> Dict[str, float]:
    """
    Extract continuous elemental properties for one atom.
    """

    element = Element(
        symbol
    )

    return {
        "atomic_number":
            safe_float(
                element.Z
            ),

        "atomic_mass":
            safe_float(
                element.atomic_mass
            ),

        "electronegativity":
            safe_float(
                element.X
            ),

        "atomic_radius_A":
            safe_float(
                getattr(
                    element,
                    "atomic_radius",
                    None,
                )
            ),

        "calculated_atomic_radius_A":
            safe_float(
                getattr(
                    element,
                    "atomic_radius_calculated",
                    None,
                )
            ),

        "periodic_row":
            safe_float(
                element.row
            ),

        "periodic_group":
            safe_float(
                element.group
            ),

        "first_ionization_energy_eV":
            first_ionization_energy(
                element
            ),

        "electron_affinity_eV":
            safe_float(
                getattr(
                    element,
                    "electron_affinity",
                    None,
                )
            ),

        "valence_electrons":
            estimate_valence_electrons(
                element
            ),
    }


def build_node_feature_vector(
    symbol: str,
    element_order: Iterable[str] = GRAPH_ELEMENT_ORDER,
) -> np.ndarray:
    """
    Build the complete atomic node-feature vector.

    Parameters
    ----------
    symbol : str
        Element symbol.

    element_order
        Ordered elements used for one-hot identity encoding.

    Returns
    -------
    numpy.ndarray
        Node feature vector.
    """

    element_order = tuple(
        element_order
    )

    properties = (
        extract_node_element_properties(
            symbol
        )
    )

    continuous_features = [
        properties[
            feature_name
        ]
        for feature_name
        in CONTINUOUS_NODE_FEATURE_NAMES
    ]

    one_hot_features = [
        1.0
        if symbol == element
        else 0.0

        for element
        in element_order
    ]

    feature_vector = np.asarray(
        continuous_features
        + one_hot_features,
        dtype=np.float32,
    )

    if not np.isfinite(
        feature_vector
    ).all():

        raise ValueError(
            "Non-finite node feature detected "
            f"for element {symbol}."
        )

    return feature_vector


def build_structure_node_matrix(
    structure,
) -> np.ndarray:
    """
    Build the node-feature matrix for a Pymatgen Structure.

    Returns
    -------
    numpy.ndarray
        Matrix with shape
        ``[number_of_atoms, node_feature_dimension]``.
    """

    if len(structure) == 0:
        raise ValueError(
            "Cannot generate node features "
            "for an empty structure."
        )

    return np.vstack(
        [
            build_node_feature_vector(
                site.specie.symbol
            )
            for site in structure
        ]
    ).astype(
        np.float32
    )


def fit_node_normalization(
    structures,
) -> Dict[str, np.ndarray]:
    """
    Fit normalization statistics for continuous node descriptors.

    One-hot identity columns are intentionally excluded.

    Parameters
    ----------
    structures
        Iterable of Pymatgen Structure objects.

    Returns
    -------
    dict
        Continuous-feature mean and standard deviation.
    """

    feature_rows = []

    for structure in structures:

        feature_matrix = (
            build_structure_node_matrix(
                structure
            )
        )

        feature_rows.append(
            feature_matrix
        )

    if not feature_rows:
        raise ValueError(
            "At least one structure is required "
            "to fit node normalization."
        )

    combined_features = np.vstack(
        feature_rows
    )

    continuous_block = combined_features[
        :,
        :CONTINUOUS_NODE_FEATURE_COUNT
    ]

    mean = continuous_block.mean(
        axis=0
    )

    std = continuous_block.std(
        axis=0
    )

    std[
        std < 1e-8
    ] = 1.0

    return {
        "mean":
            mean.astype(
                np.float32
            ),

        "std":
            std.astype(
                np.float32
            ),
    }


def normalize_node_feature_vector(
    feature_vector: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """
    Normalize continuous node features while preserving one-hot
    identity columns.
    """

    feature_vector = np.asarray(
        feature_vector,
        dtype=np.float32,
    ).copy()

    mean = np.asarray(
        mean,
        dtype=np.float32,
    )

    std = np.asarray(
        std,
        dtype=np.float32,
    )

    if len(mean) != (
        CONTINUOUS_NODE_FEATURE_COUNT
    ):

        raise ValueError(
            "Node normalization mean has "
            "incorrect dimension."
        )

    if len(std) != (
        CONTINUOUS_NODE_FEATURE_COUNT
    ):

        raise ValueError(
            "Node normalization standard deviation "
            "has incorrect dimension."
        )

    feature_vector[
        :CONTINUOUS_NODE_FEATURE_COUNT
    ] = (
        feature_vector[
            :CONTINUOUS_NODE_FEATURE_COUNT
        ]
        - mean
    ) / std

    return feature_vector


def normalize_structure_node_matrix(
    structure,
    normalization_statistics: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    Generate and normalize the complete node matrix of a structure.
    """

    matrix = build_structure_node_matrix(
        structure
    )

    mean = normalization_statistics[
        "mean"
    ]

    std = normalization_statistics[
        "std"
    ]

    normalized = matrix.copy()

    normalized[
        :,
        :CONTINUOUS_NODE_FEATURE_COUNT
    ] = (
        normalized[
            :,
            :CONTINUOUS_NODE_FEATURE_COUNT
        ]
        - mean.reshape(
            1,
            -1,
        )
    ) / std.reshape(
        1,
        -1,
    )

    if not np.isfinite(
        normalized
    ).all():

        raise ValueError(
            "Non-finite values detected after "
            "node normalization."
        )

    return normalized.astype(
        np.float32
    )


def build_node_reference_table(
    elements: Iterable[str] = GRAPH_ELEMENT_ORDER,
) -> pd.DataFrame:
    """
    Build a human-readable reference table for node features.
    """

    records = []

    for symbol in elements:

        feature_vector = (
            build_node_feature_vector(
                symbol
            )
        )

        records.append(
            {
                "element":
                    symbol,

                **dict(
                    zip(
                        NODE_FEATURE_NAMES,
                        feature_vector,
                    )
                ),
            }
        )

    return pd.DataFrame(
        records
    )


def validate_node_matrix(
    node_matrix: np.ndarray,
    expected_feature_dimension: int | None = None,
) -> bool:
    """
    Validate a graph node-feature matrix.
    """

    node_matrix = np.asarray(
        node_matrix
    )

    if node_matrix.ndim != 2:

        raise ValueError(
            "Node-feature matrix must be two-dimensional."
        )

    if node_matrix.shape[0] == 0:

        raise ValueError(
            "Node-feature matrix contains no atoms."
        )

    if expected_feature_dimension is None:

        expected_feature_dimension = len(
            NODE_FEATURE_NAMES
        )

    if node_matrix.shape[1] != (
        expected_feature_dimension
    ):

        raise ValueError(
            "Unexpected node-feature dimension: "
            f"{node_matrix.shape[1]}, expected "
            f"{expected_feature_dimension}."
        )

    if not np.isfinite(
        node_matrix
    ).all():

        raise ValueError(
            "Node-feature matrix contains "
            "non-finite values."
        )

    return True
