"""
Substitutional dopant generation utilities for GaN-DefectML.

Provides helpers for constructing the primary substitutional dopants
considered in this project:

- Mg_Ga
- Si_Ga
- O_N
- C_N

The original host-site coordinates are retained as defect-center
metadata for later structural analysis, feature engineering, and XAI.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


DOPANT_DEFINITIONS = {
    "Mg_Ga": {
        "host_species": "Ga",
        "dopant_species": "Mg",
    },
    "Si_Ga": {
        "host_species": "Ga",
        "dopant_species": "Si",
    },
    "O_N": {
        "host_species": "N",
        "dopant_species": "O",
    },
    "C_N": {
        "host_species": "N",
        "dopant_species": "C",
    },
}


def create_substitutional_dopant(
    pristine_structure,
    site_index: int,
    dopant_species: str,
) -> Tuple[object, Dict]:
    """
    Replace one host atom with a substitutional dopant.

    Parameters
    ----------
    pristine_structure
        Pymatgen Structure representing pristine GaN.

    site_index : int
        Index of the host atom to replace.

    dopant_species : str
        Element symbol of the dopant.

    Returns
    -------
    tuple
        ``(doped_structure, metadata)``
    """

    if not 0 <= site_index < len(pristine_structure):
        raise IndexError(
            f"site_index={site_index} is outside the structure "
            f"containing {len(pristine_structure)} sites."
        )

    original_site = pristine_structure[site_index]

    host_species = original_site.specie.symbol

    fractional_coordinates = np.asarray(
        original_site.frac_coords,
        dtype=float,
    ).copy()

    cartesian_coordinates = np.asarray(
        original_site.coords,
        dtype=float,
    ).copy()

    doped_structure = pristine_structure.copy()

    doped_structure.replace(
        site_index,
        dopant_species,
    )

    metadata = {
        "site_index": int(site_index),
        "host_species": host_species,
        "dopant_species": dopant_species,
        "defect_species": dopant_species,
        "defect_center_fractional":
            fractional_coordinates,
        "defect_center_cartesian":
            cartesian_coordinates,
        "original_number_of_sites":
            int(len(pristine_structure)),
        "defective_number_of_sites":
            int(len(doped_structure)),
    }

    return doped_structure, metadata


def create_named_dopant(
    pristine_structure,
    configuration_id: str,
    ga_site_index: int,
    n_site_index: int,
):
    """
    Generate one named substitutional dopant configuration.

    Parameters
    ----------
    pristine_structure
        Pristine GaN supercell.

    configuration_id : str
        One of ``Mg_Ga``, ``Si_Ga``, ``O_N``, or ``C_N``.

    ga_site_index : int
        Representative Ga site index.

    n_site_index : int
        Representative N site index.

    Returns
    -------
    tuple
        Doped structure and metadata.
    """

    if configuration_id not in DOPANT_DEFINITIONS:
        raise ValueError(
            f"Unsupported dopant configuration: "
            f"{configuration_id}"
        )

    definition = DOPANT_DEFINITIONS[
        configuration_id
    ]

    host_species = definition[
        "host_species"
    ]

    dopant_species = definition[
        "dopant_species"
    ]

    if host_species == "Ga":
        selected_site_index = ga_site_index
    elif host_species == "N":
        selected_site_index = n_site_index
    else:
        raise ValueError(
            f"Unsupported host species: {host_species}"
        )

    actual_host_species = (
        pristine_structure[
            selected_site_index
        ].specie.symbol
    )

    if actual_host_species != host_species:
        raise ValueError(
            f"Selected site {selected_site_index} is "
            f"{actual_host_species}, expected {host_species}."
        )

    structure, metadata = (
        create_substitutional_dopant(
            pristine_structure=
                pristine_structure,
            site_index=
                selected_site_index,
            dopant_species=
                dopant_species,
        )
    )

    metadata[
        "configuration_id"
    ] = configuration_id

    metadata[
        "defect_type"
    ] = "substitutional_dopant"

    return structure, metadata


def create_mg_on_ga(
    pristine_structure,
    ga_site_index: int,
):
    """
    Generate Mg substitution on a Ga site, Mg_Ga.
    """

    return create_named_dopant(
        pristine_structure=
            pristine_structure,
        configuration_id="Mg_Ga",
        ga_site_index=
            ga_site_index,
        n_site_index=-1,
    )


def create_si_on_ga(
    pristine_structure,
    ga_site_index: int,
):
    """
    Generate Si substitution on a Ga site, Si_Ga.
    """

    return create_named_dopant(
        pristine_structure=
            pristine_structure,
        configuration_id="Si_Ga",
        ga_site_index=
            ga_site_index,
        n_site_index=-1,
    )


def create_o_on_n(
    pristine_structure,
    n_site_index: int,
):
    """
    Generate O substitution on an N site, O_N.
    """

    return create_named_dopant(
        pristine_structure=
            pristine_structure,
        configuration_id="O_N",
        ga_site_index=-1,
        n_site_index=
            n_site_index,
    )


def create_c_on_n(
    pristine_structure,
    n_site_index: int,
):
    """
    Generate C substitution on an N site, C_N.
    """

    return create_named_dopant(
        pristine_structure=
            pristine_structure,
        configuration_id="C_N",
        ga_site_index=-1,
        n_site_index=
            n_site_index,
    )


def generate_primary_dopants(
    pristine_structure,
    ga_site_index: int,
    n_site_index: int,
) -> dict:
    """
    Generate the four substitutional dopants used in the project.

    Returns
    -------
    dict
        Dictionary containing Mg_Ga, Si_Ga, O_N, and C_N.
    """

    generated = {}

    for configuration_id in (
        "Mg_Ga",
        "Si_Ga",
        "O_N",
        "C_N",
    ):

        structure, metadata = (
            create_named_dopant(
                pristine_structure=
                    pristine_structure,
                configuration_id=
                    configuration_id,
                ga_site_index=
                    ga_site_index,
                n_site_index=
                    n_site_index,
            )
        )

        generated[
            configuration_id
        ] = {
            "structure":
                structure,
            "metadata":
                metadata,
        }

    return generated


def validate_substitutional_dopant(
    pristine_structure,
    doped_structure,
    host_species: str,
    dopant_species: str,
) -> bool:
    """
    Validate the composition change caused by substitutional doping.

    Expected behavior:
    - total number of atoms unchanged,
    - host species decreases by one,
    - dopant species increases by one.

    Raises
    ------
    ValueError
        If the expected composition change is not observed.
    """

    if len(doped_structure) != len(
        pristine_structure
    ):
        raise ValueError(
            "Substitutional doping must preserve "
            "the total atom count."
        )

    pristine_host_count = sum(
        site.specie.symbol
        == host_species
        for site in pristine_structure
    )

    doped_host_count = sum(
        site.specie.symbol
        == host_species
        for site in doped_structure
    )

    pristine_dopant_count = sum(
        site.specie.symbol
        == dopant_species
        for site in pristine_structure
    )

    doped_dopant_count = sum(
        site.specie.symbol
        == dopant_species
        for site in doped_structure
    )

    if doped_host_count != (
        pristine_host_count - 1
    ):
        raise ValueError(
            f"Expected one fewer {host_species} atom."
        )

    if doped_dopant_count != (
        pristine_dopant_count + 1
    ):
        raise ValueError(
            f"Expected one additional {dopant_species} atom."
        )

    return True
