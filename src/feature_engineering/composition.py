"""
Composition descriptor utilities for GaN-DefectML.

This module generates global composition features from pristine and
defective GaN structures using elemental properties from Pymatgen.
"""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd

from pymatgen.core import Element


DEFAULT_TRACKED_ELEMENTS = (
    "Ga",
    "N",
    "Mg",
    "Si",
    "O",
    "C",
)


def safe_float(value) -> float:
    """
    Convert a scalar-like value to float.

    Returns NaN when conversion is not possible.
    """

    if value is None:
        return np.nan

    try:
        return float(value)

    except (TypeError, ValueError):
        return np.nan


def get_element_properties(
    symbol: str,
) -> Dict[str, float]:
    """
    Return elemental properties used for composition descriptors.

    Parameters
    ----------
    symbol : str
        Chemical element symbol.

    Returns
    -------
    dict
        Selected elemental properties.
    """

    element = Element(symbol)

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
                element.atomic_radius
            ),

        "row":
            safe_float(
                element.row
            ),

        "group":
            safe_float(
                element.group
            ),
    }


def weighted_property_statistics(
    property_values,
    weights,
) -> Dict[str, float]:
    """
    Calculate weighted elemental-property statistics.

    Returns
    -------
    dict
        Weighted mean, weighted standard deviation, minimum,
        maximum, and range.
    """

    property_values = np.asarray(
        property_values,
        dtype=float,
    )

    weights = np.asarray(
        weights,
        dtype=float,
    )

    valid_mask = np.isfinite(
        property_values
    )

    if not valid_mask.any():

        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "range": np.nan,
        }

    valid_values = property_values[
        valid_mask
    ]

    valid_weights = weights[
        valid_mask
    ]

    normalized_weights = (
        valid_weights
        / valid_weights.sum()
    )

    weighted_mean = np.sum(
        normalized_weights
        * valid_values
    )

    weighted_variance = np.sum(
        normalized_weights
        * (
            valid_values
            - weighted_mean
        ) ** 2
    )

    minimum = float(
        np.min(
            valid_values
        )
    )

    maximum = float(
        np.max(
            valid_values
        )
    )

    return {
        "mean":
            float(
                weighted_mean
            ),

        "std":
            float(
                np.sqrt(
                    weighted_variance
                )
            ),

        "min":
            minimum,

        "max":
            maximum,

        "range":
            float(
                maximum
                - minimum
            ),
    }


def generate_composition_descriptors(
    structure,
    tracked_elements: Iterable[str] = DEFAULT_TRACKED_ELEMENTS,
) -> Dict[str, float]:
    """
    Generate global composition descriptors for one structure.

    Parameters
    ----------
    structure
        Pymatgen Structure.

    tracked_elements
        Elements for which explicit fractions and counts are stored.

    Returns
    -------
    dict
        Composition descriptor dictionary.
    """

    composition = (
        structure.composition
    )

    element_amounts = (
        composition.get_el_amt_dict()
    )

    total_atoms = float(
        sum(
            element_amounts.values()
        )
    )

    property_names = [
        "atomic_number",
        "atomic_mass",
        "electronegativity",
        "atomic_radius_A",
        "row",
        "group",
    ]

    property_values = {
        property_name: []
        for property_name
        in property_names
    }

    weights = []

    for symbol, amount in (
        element_amounts.items()
    ):

        properties = get_element_properties(
            symbol
        )

        weights.append(
            float(amount)
        )

        for property_name in property_names:

            property_values[
                property_name
            ].append(
                properties[
                    property_name
                ]
            )

    descriptors = {
        "number_of_elements":
            int(
                len(
                    element_amounts
                )
            ),

        "total_atoms_composition":
            total_atoms,
    }

    for property_name in property_names:

        statistics = (
            weighted_property_statistics(
                property_values=
                    property_values[
                        property_name
                    ],
                weights=
                    weights,
            )
        )

        descriptors[
            f"mean_{property_name}"
        ] = statistics[
            "mean"
        ]

        descriptors[
            f"std_{property_name}"
        ] = statistics[
            "std"
        ]

        descriptors[
            f"min_{property_name}"
        ] = statistics[
            "min"
        ]

        descriptors[
            f"max_{property_name}"
        ] = statistics[
            "max"
        ]

        descriptors[
            f"range_{property_name}"
        ] = statistics[
            "range"
        ]

    for symbol in tracked_elements:

        amount = float(
            element_amounts.get(
                symbol,
                0.0,
            )
        )

        descriptors[
            f"count_{symbol}"
        ] = amount

        descriptors[
            f"fraction_{symbol}"
        ] = (
            amount / total_atoms
            if total_atoms > 0
            else np.nan
        )

    return descriptors


def build_composition_feature_dataframe(
    structure_dataframe: pd.DataFrame,
    structure_column: str = "structure_object",
    configuration_id_column: str = "configuration_id",
    tracked_elements: Iterable[str] = DEFAULT_TRACKED_ELEMENTS,
) -> pd.DataFrame:
    """
    Generate composition descriptors for a structure library.

    Parameters
    ----------
    structure_dataframe : pandas.DataFrame
        Table containing configuration IDs and Pymatgen structures.

    structure_column : str, default="structure_object"
        Column containing structures.

    configuration_id_column : str, default="configuration_id"
        Structure identifier column.

    tracked_elements
        Explicit elements for count/fraction features.

    Returns
    -------
    pandas.DataFrame
        Composition descriptor table.
    """

    records = []

    for _, row in (
        structure_dataframe.iterrows()
    ):

        configuration_id = row[
            configuration_id_column
        ]

        structure = row[
            structure_column
        ]

        descriptors = (
            generate_composition_descriptors(
                structure=
                    structure,
                tracked_elements=
                    tracked_elements,
            )
        )

        records.append(
            {
                "configuration_id":
                    configuration_id,

                **descriptors,
            }
        )

    return pd.DataFrame(
        records
    )


def validate_composition_features(
    feature_dataframe: pd.DataFrame,
    tracked_elements: Iterable[str] = DEFAULT_TRACKED_ELEMENTS,
    tolerance: float = 1e-6,
) -> bool:
    """
    Validate explicit elemental fractions in a composition table.

    The tracked fractions are not necessarily expected to sum to one
    for an arbitrary material system. For the present GaN defect
    library, however, all species are included in the tracked set.

    Raises
    ------
    ValueError
        If the tracked fractions do not sum to approximately one.
    """

    fraction_columns = [
        f"fraction_{symbol}"
        for symbol in tracked_elements
        if f"fraction_{symbol}"
        in feature_dataframe.columns
    ]

    if not fraction_columns:
        raise ValueError(
            "No tracked fraction columns were found."
        )

    fraction_sum = (
        feature_dataframe[
            fraction_columns
        ]
        .sum(axis=1)
    )

    invalid_mask = (
        np.abs(
            fraction_sum
            - 1.0
        ) > tolerance
    )

    if invalid_mask.any():

        invalid_ids = (
            feature_dataframe.loc[
                invalid_mask,
                "configuration_id",
            ]
            .tolist()
        )

        raise ValueError(
            "Element fractions do not sum to one "
            f"for configurations: {invalid_ids}"
        )

    return True
