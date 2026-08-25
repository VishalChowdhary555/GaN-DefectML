"""
Vacancy-generation utilities for GaN-DefectML.

Provides functions for generating intrinsic gallium and nitrogen
vacancies from a pristine GaN supercell while retaining metadata
about the removed atomic site.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def create_vacancy(
    pristine_structure,
    site_index: int,
) -> Tuple[object, Dict]:
    """
    Create a vacancy by removing one atomic site.

    Parameters
    ----------
    pristine_structure
        Pymatgen Structure representing pristine GaN.

    site_index : int
        Index of the atom to remove.

    Returns
    -------
    tuple
        ``(defect_structure, metadata)``

        The metadata retains the removed species and its original
        coordinates so the vacancy center remains available after
        the atom itself has been deleted.
    """

    if not 0 <= site_index < len(
        pristine_structure
    ):
        raise IndexError(
            f"site_index={site_index} is outside "
            f"the structure containing "
            f"{len(pristine_structure)} sites."
        )

    original_site = pristine_structure[
        site_index
    ]

    removed_species = (
        original_site.specie.symbol
    )

    fractional_coordinates = np.asarray(
        original_site.frac_coords,
        dtype=float,
    ).copy()

    cartesian_coordinates = np.asarray(
        original_site.coords,
        dtype=float,
    ).copy()

    defect_structure = (
        pristine_structure.copy()
    )

    defect_structure.remove_sites(
        [site_index]
    )

    metadata = {
        "removed_site_index":
            int(site_index),

        "removed_species":
            removed_species,

        "defect_center_fractional":
            fractional_coordinates,

        "defect_center_cartesian":
            cartesian_coordinates,

        "original_number_of_sites":
            int(len(pristine_structure)),

        "defective_number_of_sites":
            int(len(defect_structure)),
    }

    return (
        defect_structure,
        metadata,
    )


def create_ga_vacancy(
    pristine_structure,
    ga_site_index: int,
):
    """
    Generate a gallium vacancy, V_Ga.

    Parameters
    ----------
    pristine_structure
        Pristine GaN supercell.

    ga_site_index : int
        Index of the Ga atom to remove.

    Returns
    -------
    tuple
        Vacancy structure and associated metadata.
    """

    site = pristine_structure[
        ga_site_index
    ]

    if site.specie.symbol != "Ga":
        raise ValueError(
            f"Site {ga_site_index} is "
            f"{site.specie.symbol}, not Ga."
        )

    structure, metadata = create_vacancy(
        pristine_structure=
            pristine_structure,
        site_index=ga_site_index,
    )

    metadata[
        "configuration_id"
    ] = "V_Ga"

    metadata[
        "defect_type"
    ] = "vacancy"

    metadata[
        "host_species"
    ] = "Ga"

    return structure, metadata


def create_n_vacancy(
    pristine_structure,
    n_site_index: int,
):
    """
    Generate a nitrogen vacancy, V_N.

    Parameters
    ----------
    pristine_structure
        Pristine GaN supercell.

    n_site_index : int
        Index of the N atom to remove.

    Returns
    -------
    tuple
        Vacancy structure and associated metadata.
    """

    site = pristine_structure[
        n_site_index
    ]

    if site.specie.symbol != "N":
        raise ValueError(
            f"Site {n_site_index} is "
            f"{site.specie.symbol}, not N."
        )

    structure, metadata = create_vacancy(
        pristine_structure=
            pristine_structure,
        site_index=n_site_index,
    )

    metadata[
        "configuration_id"
    ] = "V_N"

    metadata[
        "defect_type"
    ] = "vacancy"

    metadata[
        "host_species"
    ] = "N"

    return structure, metadata


def generate_intrinsic_vacancies(
    pristine_structure,
    ga_site_index: int,
    n_site_index: int,
) -> dict:
    """
    Generate both intrinsic GaN vacancy configurations.

    Returns
    -------
    dict
        Dictionary containing ``V_Ga`` and ``V_N`` structures
        together with their metadata.
    """

    ga_vacancy, ga_metadata = (
        create_ga_vacancy(
            pristine_structure=
                pristine_structure,
            ga_site_index=
                ga_site_index,
        )
    )

    n_vacancy, n_metadata = (
        create_n_vacancy(
            pristine_structure=
                pristine_structure,
            n_site_index=
                n_site_index,
        )
    )

    return {
        "V_Ga": {
            "structure":
                ga_vacancy,

            "metadata":
                ga_metadata,
        },

        "V_N": {
            "structure":
                n_vacancy,

            "metadata":
                n_metadata,
        },
    }


def validate_vacancy_structure(
    pristine_structure,
    vacancy_structure,
    removed_species: str,
) -> bool:
    """
    Validate basic stoichiometric behavior of a vacancy.

    The defective structure must contain exactly one fewer atom
    than the pristine structure and exactly one fewer atom of the
    removed species.

    Raises
    ------
    ValueError
        If validation fails.
    """

    if (
        len(vacancy_structure)
        != len(pristine_structure) - 1
    ):
        raise ValueError(
            "Vacancy structure must contain exactly "
            "one fewer atom than the pristine structure."
        )

    pristine_count = sum(
        site.specie.symbol
        == removed_species
        for site in pristine_structure
    )

    vacancy_count = sum(
        site.specie.symbol
        == removed_species
        for site in vacancy_structure
    )

    if vacancy_count != (
        pristine_count - 1
    ):
        raise ValueError(
            f"Expected one fewer {removed_species} atom, "
            f"but pristine={pristine_count} and "
            f"vacancy={vacancy_count}."
        )

    return True
