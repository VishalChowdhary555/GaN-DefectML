"""
Supercell construction utilities for GaN-DefectML.

This module provides helpers for:
- selecting a pristine host structure,
- constructing defect supercells,
- counting species,
- locating representative central Ga and N sites,
- and summarizing supercell composition.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np


def get_species_count(
    structure,
    element_symbol: str,
) -> int:
    """
    Count atoms of a selected element in a Pymatgen Structure.

    Parameters
    ----------
    structure
        Pymatgen Structure.

    element_symbol : str
        Element symbol, e.g. ``"Ga"`` or ``"N"``.

    Returns
    -------
    int
        Number of atoms matching the selected element.
    """

    return int(
        sum(
            site.specie.symbol == element_symbol
            for site in structure
        )
    )


def get_species_indices(
    structure,
    element_symbol: str,
) -> list[int]:
    """
    Return indices of all atoms matching a selected element.
    """

    return [
        index
        for index, site in enumerate(structure)
        if site.specie.symbol == element_symbol
    ]


def build_supercell(
    primitive_structure,
    replication_matrix: Iterable[int] = (3, 3, 2),
):
    """
    Construct a replicated GaN supercell.

    Parameters
    ----------
    primitive_structure
        Pymatgen Structure representing the pristine host.

    replication_matrix : iterable of int, default=(3, 3, 2)
        Supercell replication along the lattice vectors.

    Returns
    -------
    pymatgen.core.structure.Structure
        Replicated supercell.
    """

    replication_matrix = list(
        replication_matrix
    )

    if len(replication_matrix) != 3:
        raise ValueError(
            "replication_matrix must contain exactly "
            "three integers."
        )

    if any(
        int(value) <= 0
        for value in replication_matrix
    ):
        raise ValueError(
            "All supercell replication values must be positive."
        )

    supercell = primitive_structure.copy()

    supercell.make_supercell(
        replication_matrix
    )

    return supercell


def summarize_supercell(
    structure,
    replication_matrix: Iterable[int] | None = None,
) -> Dict[str, float]:
    """
    Summarize a GaN supercell.

    Returns
    -------
    dict
        Site counts, density, volume, and defect concentration scale.
    """

    total_atoms = len(structure)

    ga_atoms = get_species_count(
        structure,
        "Ga",
    )

    n_atoms = get_species_count(
        structure,
        "N",
    )

    summary = {
        "total_atoms":
            total_atoms,

        "Ga_atoms":
            ga_atoms,

        "N_atoms":
            n_atoms,

        "supercell_volume_A3":
            float(structure.volume),

        "supercell_density_g_cm3":
            float(structure.density),

        "single_defect_fraction_all_atoms_percent":
            (
                100.0 / total_atoms
                if total_atoms > 0
                else np.nan
            ),
    }

    if replication_matrix is not None:
        summary[
            "replication_matrix"
        ] = list(replication_matrix)

    return summary


def periodic_fractional_distance(
    lattice,
    fractional_a,
    fractional_b,
) -> float:
    """
    Calculate minimum-image distance between fractional coordinates.
    """

    distance, _ = lattice.get_distance_and_image(
        fractional_a,
        fractional_b,
    )

    return float(distance)


def find_nearest_site_to_fractional_point(
    structure,
    element_symbol: str,
    target_fractional=(0.5, 0.5, 0.5),
) -> int:
    """
    Find the atom of a selected element closest to a target point.

    The function is useful for selecting representative defect sites
    near the center of a periodic supercell.

    Parameters
    ----------
    structure
        Pymatgen Structure.

    element_symbol : str
        Species to search for.

    target_fractional : tuple, default=(0.5, 0.5, 0.5)
        Target fractional coordinate.

    Returns
    -------
    int
        Index of the closest matching site.
    """

    candidate_indices = get_species_indices(
        structure,
        element_symbol,
    )

    if not candidate_indices:
        raise ValueError(
            f"No {element_symbol} atoms were found "
            "in the supplied structure."
        )

    target_fractional = np.asarray(
        target_fractional,
        dtype=float,
    )

    distances = []

    for index in candidate_indices:

        distance = periodic_fractional_distance(
            lattice=structure.lattice,
            fractional_a=
                structure[index].frac_coords,
            fractional_b=
                target_fractional,
        )

        distances.append(
            (
                distance,
                index,
            )
        )

    distances.sort(
        key=lambda item: item[0]
    )

    return int(
        distances[0][1]
    )


def select_representative_ga_n_sites(
    structure,
    target_fractional=(0.5, 0.5, 0.5),
) -> Tuple[int, int]:
    """
    Select representative Ga and N sites near the supercell center.

    Returns
    -------
    tuple of int
        ``(ga_site_index, n_site_index)``
    """

    ga_index = (
        find_nearest_site_to_fractional_point(
            structure=structure,
            element_symbol="Ga",
            target_fractional=
                target_fractional,
        )
    )

    n_index = (
        find_nearest_site_to_fractional_point(
            structure=structure,
            element_symbol="N",
            target_fractional=
                target_fractional,
        )
    )

    return (
        ga_index,
        n_index,
    )


def get_representative_site_metadata(
    structure,
    site_index: int,
) -> dict:
    """
    Return metadata for a selected representative atomic site.
    """

    site = structure[
        site_index
    ]

    return {
        "site_index":
            int(site_index),

        "species":
            site.specie.symbol,

        "fractional_coordinates":
            np.asarray(
                site.frac_coords,
                dtype=float,
            ),

        "cartesian_coordinates":
            np.asarray(
                site.coords,
                dtype=float,
            ),
    }


def validate_gan_supercell(
    structure,
    require_equal_sublattices: bool = True,
) -> bool:
    """
    Validate basic composition constraints for pristine GaN.

    Parameters
    ----------
    structure
        Candidate GaN supercell.

    require_equal_sublattices : bool, default=True
        Require equal Ga and N counts.

    Returns
    -------
    bool
        True when validation succeeds.

    Raises
    ------
    ValueError
        If the supercell does not satisfy the expected composition.
    """

    ga_count = get_species_count(
        structure,
        "Ga",
    )

    n_count = get_species_count(
        structure,
        "N",
    )

    if ga_count == 0 or n_count == 0:
        raise ValueError(
            "Structure must contain both Ga and N."
        )

    if (
        require_equal_sublattices
        and ga_count != n_count
    ):
        raise ValueError(
            "Pristine GaN supercell must contain equal "
            f"Ga and N counts, found Ga={ga_count}, N={n_count}."
        )

    return True
