"""
Antisite-defect generation utilities for GaN-DefectML.

Provides functions for constructing:
- Ga on N antisites, Ga_N
- N on Ga antisites, N_Ga

The original host-site coordinates are retained as defect-center metadata.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def create_substitution(
    pristine_structure,
    site_index: int,
    new_species: str,
) -> Tuple[object, Dict]:
    """
    Replace one atomic species with another at a selected lattice site.

    Parameters
    ----------
    pristine_structure
        Pymatgen Structure representing pristine GaN.

    site_index : int
        Index of the atom to replace.

    new_species : str
        Element symbol to insert.

    Returns
    -------
    tuple
        ``(defect_structure, metadata)``
    """

    if not 0 <= site_index < len(pristine_structure):
        raise IndexError(
            f"site_index={site_index} is outside the structure "
            f"containing {len(pristine_structure)} sites."
        )

    original_site = pristine_structure[site_index]

    original_species = original_site.specie.symbol

    fractional_coordinates = np.asarray(
        original_site.frac_coords,
        dtype=float,
    ).copy()

    cartesian_coordinates = np.asarray(
        original_site.coords,
        dtype=float,
    ).copy()

    defect_structure = pristine_structure.copy()

    defect_structure.replace(
        site_index,
        new_species,
    )

    metadata = {
        "site_index": int(site_index),
        "host_species": original_species,
        "defect_species": new_species,
        "defect_center_fractional":
            fractional_coordinates,
        "defect_center_cartesian":
            cartesian_coordinates,
        "original_number_of_sites":
            int(len(pristine_structure)),
        "defective_number_of_sites":
            int(len(defect_structure)),
    }

    return defect_structure, metadata


def create_ga_on_n_antisite(
    pristine_structure,
    n_site_index: int,
):
    """
    Generate a Ga-on-N antisite defect, Ga_N.
    """

    site = pristine_structure[n_site_index]

    if site.specie.symbol != "N":
        raise ValueError(
            f"Site {n_site_index} is "
            f"{site.specie.symbol}, not N."
        )

    structure, metadata = create_substitution(
        pristine_structure=
            pristine_structure,
        site_index=n_site_index,
        new_species="Ga",
    )

    metadata["configuration_id"] = "Ga_N"
    metadata["defect_type"] = "antisite"

    return structure, metadata


def create_n_on_ga_antisite(
    pristine_structure,
    ga_site_index: int,
):
    """
    Generate an N-on-Ga antisite defect, N_Ga.
    """

    site = pristine_structure[ga_site_index]

    if site.specie.symbol != "Ga":
        raise ValueError(
            f"Site {ga_site_index} is "
            f"{site.specie.symbol}, not Ga."
        )

    structure, metadata = create_substitution(
        pristine_structure=
            pristine_structure,
        site_index=ga_site_index,
        new_species="N",
    )

    metadata["configuration_id"] = "N_Ga"
    metadata["defect_type"] = "antisite"

    return structure, metadata


def generate_intrinsic_antisites(
    pristine_structure,
    ga_site_index: int,
    n_site_index: int,
) -> dict:
    """
    Generate both intrinsic GaN antisite configurations.

    Returns
    -------
    dict
        Dictionary containing ``Ga_N`` and ``N_Ga``.
    """

    ga_on_n, ga_on_n_metadata = (
        create_ga_on_n_antisite(
            pristine_structure=
                pristine_structure,
            n_site_index=
                n_site_index,
        )
    )

    n_on_ga, n_on_ga_metadata = (
        create_n_on_ga_antisite(
            pristine_structure=
                pristine_structure,
            ga_site_index=
                ga_site_index,
        )
    )

    return {
        "Ga_N": {
            "structure":
                ga_on_n,

            "metadata":
                ga_on_n_metadata,
        },

        "N_Ga": {
            "structure":
                n_on_ga,

            "metadata":
                n_on_ga_metadata,
        },
    }


def validate_antisite_structure(
    pristine_structure,
    antisite_structure,
    host_species: str,
    defect_species: str,
) -> bool:
    """
    Validate stoichiometric changes caused by an antisite.

    The total atom count must remain unchanged. The host species
    count decreases by one and the defect species count increases
    by one.

    Raises
    ------
    ValueError
        If the expected composition change is not observed.
    """

    if len(antisite_structure) != len(
        pristine_structure
    ):
        raise ValueError(
            "Antisite substitution must not change "
            "the total number of atoms."
        )

    pristine_host_count = sum(
        site.specie.symbol == host_species
        for site in pristine_structure
    )

    defect_host_count = sum(
        site.specie.symbol == host_species
        for site in antisite_structure
    )

    pristine_defect_count = sum(
        site.specie.symbol == defect_species
        for site in pristine_structure
    )

    antisite_defect_count = sum(
        site.specie.symbol == defect_species
        for site in antisite_structure
    )

    if defect_host_count != (
        pristine_host_count - 1
    ):
        raise ValueError(
            f"Expected one fewer {host_species} atom."
        )

    if antisite_defect_count != (
        pristine_defect_count + 1
    ):
        raise ValueError(
            f"Expected one additional {defect_species} atom."
        )

    return True
