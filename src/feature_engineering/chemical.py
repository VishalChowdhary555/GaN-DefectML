"""
Chemical perturbation descriptor utilities for GaN-DefectML.

This module generates physics-informed descriptors describing the
chemical change introduced by vacancies, antisites, substitutional
dopants, and interstitials.

The central idea is:

    delta_property = final_site_property - initial_site_property

where an empty site is represented by a zero-valued property vector.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from pymatgen.core import Element


DEFAULT_CHEMICAL_PROPERTIES = (
    "atomic_number",
    "atomic_mass",
    "electronegativity",
    "atomic_radius_A",
    "calculated_atomic_radius_A",
    "row",
    "group",
    "first_ionization_energy_eV",
    "electron_affinity_eV",
    "valence_electrons",
)


def safe_numeric_property(
    value,
) -> float:
    """
    Convert an elemental property to float.

    Missing or non-numeric values are returned as NaN.
    """

    if value is None:
        return np.nan

    try:
        return float(value)

    except (TypeError, ValueError):
        return np.nan


def get_first_ionization_energy(
    element: Element,
) -> float:
    """
    Return the first ionization energy in eV when available.
    """

    scalar_value = getattr(
        element,
        "ionization_energy",
        None,
    )

    scalar_value = safe_numeric_property(
        scalar_value
    )

    if np.isfinite(scalar_value):
        return scalar_value

    values = getattr(
        element,
        "ionization_energies",
        None,
    )

    try:
        if values is not None and len(values) > 0:
            return float(values[0])

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
    Estimate conventional valence-electron count from group number.

    For main-group elements:
    - groups 1-2  -> 1-2
    - groups 13-18 -> 3-8

    For transition metals, the group number is retained as a simple
    numerical descriptor.

    This is a descriptor convention, not an oxidation-state assignment.
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


def extract_elemental_properties(
    symbol: str,
) -> Dict[str, float]:
    """
    Extract elemental properties used for chemical perturbation
    descriptors.
    """

    element = Element(symbol)

    return {
        "atomic_number":
            safe_numeric_property(
                element.Z
            ),

        "atomic_mass":
            safe_numeric_property(
                element.atomic_mass
            ),

        "electronegativity":
            safe_numeric_property(
                element.X
            ),

        "atomic_radius_A":
            safe_numeric_property(
                getattr(
                    element,
                    "atomic_radius",
                    None,
                )
            ),

        "calculated_atomic_radius_A":
            safe_numeric_property(
                getattr(
                    element,
                    "atomic_radius_calculated",
                    None,
                )
            ),

        "row":
            safe_numeric_property(
                element.row
            ),

        "group":
            safe_numeric_property(
                element.group
            ),

        "first_ionization_energy_eV":
            get_first_ionization_energy(
                element
            ),

        "electron_affinity_eV":
            safe_numeric_property(
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


def empty_site_properties(
    property_names: Iterable[str] = DEFAULT_CHEMICAL_PROPERTIES,
) -> Dict[str, float]:
    """
    Return a zero-valued property vector representing an empty site.
    """

    return {
        property_name: 0.0
        for property_name
        in property_names
    }


def species_property_dictionary(
    symbol: Optional[str],
) -> Dict[str, float]:
    """
    Return elemental properties or the empty-site baseline.

    Parameters
    ----------
    symbol
        Element symbol. ``None`` represents an empty lattice site.
    """

    if (
        symbol is None
        or pd.isna(symbol)
    ):
        return empty_site_properties()

    return extract_elemental_properties(
        str(symbol)
    )


def determine_chemical_perturbation_pair(
    configuration_id: str,
    configuration_family: str,
    defect_species: Optional[str],
    host_species: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """
    Determine the initial and final species associated with a defect.

    Returns
    -------
    tuple
        ``(initial_species, final_species)``

    Conventions
    -----------
    pristine:
        empty -> empty

    vacancy:
        host -> empty

    antisite/substitutional dopant:
        host -> defect species

    interstitial:
        empty -> inserted species
    """

    if configuration_id == "Pristine":
        return None, None

    family = str(
        configuration_family
    )

    family_normalized = (
        family.lower()
        .replace("_", " ")
        .strip()
    )

    if "vacancy" in family_normalized:
        return host_species, None

    if (
        "antisite" in family_normalized
        or "substitutional" in family_normalized
    ):
        return (
            host_species,
            defect_species,
        )

    if "interstitial" in family_normalized:
        return (
            None,
            defect_species,
        )

    raise ValueError(
        "Unsupported configuration family "
        f"'{configuration_family}' for "
        f"{configuration_id}."
    )


def calculate_property_difference(
    initial_value: float,
    final_value: float,
) -> dict:
    """
    Calculate signed, absolute, and normalized property differences.
    """

    if not (
        np.isfinite(initial_value)
        and np.isfinite(final_value)
    ):
        return {
            "delta": np.nan,
            "absolute_delta": np.nan,
            "normalized_delta": np.nan,
            "normalized_absolute_delta": np.nan,
        }

    delta = (
        final_value
        - initial_value
    )

    absolute_delta = abs(
        delta
    )

    denominator = (
        abs(initial_value)
        + abs(final_value)
        + 1e-8
    )

    normalized_delta = (
        delta / denominator
    )

    return {
        "delta":
            float(delta),

        "absolute_delta":
            float(
                absolute_delta
            ),

        "normalized_delta":
            float(
                normalized_delta
            ),

        "normalized_absolute_delta":
            float(
                abs(
                    normalized_delta
                )
            ),
    }


def generate_chemical_perturbation_descriptors(
    configuration_id: str,
    configuration_family: str,
    defect_species: Optional[str],
    host_species: Optional[str],
) -> Dict[str, float]:
    """
    Generate the full chemical-perturbation descriptor block.

    Parameters
    ----------
    configuration_id
        Defect configuration identifier.

    configuration_family
        Pristine, Vacancy, Antisite, Substitutional dopant,
        or Interstitial.

    defect_species
        Species introduced by the defect.

    host_species
        Species replaced or removed.

    Returns
    -------
    dict
        Chemical descriptor dictionary.
    """

    (
        initial_species,
        final_species,
    ) = determine_chemical_perturbation_pair(
        configuration_id=
            configuration_id,
        configuration_family=
            configuration_family,
        defect_species=
            defect_species,
        host_species=
            host_species,
    )

    initial_properties = (
        species_property_dictionary(
            initial_species
        )
    )

    final_properties = (
        species_property_dictionary(
            final_species
        )
    )

    descriptors = {
        "initial_site_species":
            initial_species,

        "final_site_species":
            final_species,
    }

    normalized_absolute_values = []

    for property_name in (
        DEFAULT_CHEMICAL_PROPERTIES
    ):

        initial_value = (
            initial_properties[
                property_name
            ]
        )

        final_value = (
            final_properties[
                property_name
            ]
        )

        difference = (
            calculate_property_difference(
                initial_value=
                    initial_value,
                final_value=
                    final_value,
            )
        )

        descriptors[
            f"initial_{property_name}"
        ] = initial_value

        descriptors[
            f"final_{property_name}"
        ] = final_value

        descriptors[
            f"delta_{property_name}"
        ] = difference[
            "delta"
        ]

        descriptors[
            f"abs_delta_{property_name}"
        ] = difference[
            "absolute_delta"
        ]

        descriptors[
            f"normalized_delta_{property_name}"
        ] = difference[
            "normalized_delta"
        ]

        descriptors[
            f"normalized_abs_delta_{property_name}"
        ] = difference[
            "normalized_absolute_delta"
        ]

        if np.isfinite(
            difference[
                "normalized_absolute_delta"
            ]
        ):
            normalized_absolute_values.append(
                difference[
                    "normalized_absolute_delta"
                ]
            )

    descriptors[
        "combined_chemical_perturbation"
    ] = (
        float(
            np.mean(
                normalized_absolute_values
            )
        )
        if normalized_absolute_values
        else np.nan
    )

    family_lower = str(
        configuration_family
    ).lower()

    descriptors[
        "chemical_operation_none"
    ] = int(
        configuration_id
        == "Pristine"
    )

    descriptors[
        "chemical_operation_removal"
    ] = int(
        "vacancy"
        in family_lower
    )

    descriptors[
        "chemical_operation_replacement"
    ] = int(
        (
            "antisite"
            in family_lower
        )
        or (
            "substitutional"
            in family_lower
        )
    )

    descriptors[
        "chemical_operation_insertion"
    ] = int(
        "interstitial"
        in family_lower
    )

    return descriptors


def build_chemical_feature_dataframe(
    structure_library_df: pd.DataFrame,
    configuration_id_column: str = "configuration_id",
    family_column: str = "configuration_family",
    defect_species_column: str = "defect_species",
    host_species_column: str = "host_species",
) -> pd.DataFrame:
    """
    Generate chemical perturbation descriptors for a defect library.
    """

    records = []

    for _, row in (
        structure_library_df.iterrows()
    ):

        configuration_id = row[
            configuration_id_column
        ]

        descriptors = (
            generate_chemical_perturbation_descriptors(
                configuration_id=
                    configuration_id,

                configuration_family=
                    row[
                        family_column
                    ],

                defect_species=
                    row.get(
                        defect_species_column
                    ),

                host_species=
                    row.get(
                        host_species_column
                    ),
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


def build_elemental_reference_table(
    elements: Iterable[str],
) -> pd.DataFrame:
    """
    Build a reference table of elemental properties used by the project.
    """

    records = []

    for symbol in elements:

        records.append(
            {
                "element":
                    symbol,

                **extract_elemental_properties(
                    symbol
                ),
            }
        )

    return pd.DataFrame(
        records
    )


def validate_chemical_descriptors(
    chemical_feature_df: pd.DataFrame,
) -> bool:
    """
    Validate chemical descriptor consistency.

    Checks that pristine chemical perturbations are zero and that no
    infinite numerical values are present.
    """

    numeric_df = (
        chemical_feature_df
        .select_dtypes(
            include=[
                np.number,
                "bool",
            ]
        )
    )

    if np.isinf(
        numeric_df.to_numpy(
            dtype=float
        )
    ).any():
        raise ValueError(
            "Infinite values detected in chemical descriptors."
        )

    pristine_rows = (
        chemical_feature_df[
            chemical_feature_df[
                "configuration_id"
            ] == "Pristine"
        ]
    )

    if len(pristine_rows) == 1:

        pristine_row = (
            pristine_rows.iloc[0]
        )

        delta_columns = [
            column
            for column
            in chemical_feature_df.columns
            if column.startswith(
                "delta_"
            )
        ]

        for column in delta_columns:

            value = pristine_row[
                column
            ]

            if (
                pd.notna(value)
                and not np.isclose(
                    float(value),
                    0.0,
                )
            ):
                raise ValueError(
                    "Pristine configuration contains "
                    f"nonzero perturbation in {column}."
                )

    return True
