"""
Coordination analysis utilities for GaN-DefectML.

This module provides local coordination and first-shell Ga-N bond
analysis using Pymatgen's CrystalNN method.

The implementation avoids requiring explicit oxidation states, which
allows it to operate directly on Materials Project GaN structures.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from pymatgen.analysis.local_env import CrystalNN


def get_element_symbol(site) -> str:
    """
    Return the elemental symbol associated with a Pymatgen site.

    Parameters
    ----------
    site
        Pymatgen PeriodicSite object.

    Returns
    -------
    str
        Element symbol.
    """

    specie = site.specie

    if hasattr(specie, "symbol"):
        return specie.symbol

    return str(specie)


def create_crystal_nn(
    weighted_cn: bool = True,
    cation_anion: bool = False,
) -> CrystalNN:
    """
    Create a CrystalNN coordination analyzer.

    Parameters
    ----------
    weighted_cn : bool, default=True
        Return weighted coordination numbers.

    cation_anion : bool, default=False
        Whether CrystalNN should distinguish cation-anion
        interactions.

        This remains False because the Materials Project structures
        used by this project do not necessarily contain explicit
        oxidation states.

    Returns
    -------
    CrystalNN
        Configured CrystalNN analyzer.
    """

    return CrystalNN(
        weighted_cn=weighted_cn,
        cation_anion=cation_anion,
    )


def calculate_site_coordination(
    structure,
    site_index: int,
    crystal_nn: CrystalNN | None = None,
) -> float:
    """
    Calculate the coordination number of one atomic site.

    Parameters
    ----------
    structure
        Pymatgen Structure.

    site_index : int
        Index of the atomic site.

    crystal_nn : CrystalNN, optional
        Existing CrystalNN analyzer.

    Returns
    -------
    float
        Weighted coordination number.
    """

    if crystal_nn is None:
        crystal_nn = create_crystal_nn()

    coordination = crystal_nn.get_cn(
        structure,
        site_index,
        use_weights=True,
    )

    return float(coordination)


def calculate_element_coordination(
    structure,
    element_symbol: str,
    crystal_nn: CrystalNN | None = None,
) -> np.ndarray:
    """
    Calculate coordination numbers for all atoms of one element.

    Parameters
    ----------
    structure
        Pymatgen Structure.

    element_symbol : str
        Element to analyze, e.g. ``"Ga"`` or ``"N"``.

    crystal_nn : CrystalNN, optional
        Existing CrystalNN analyzer.

    Returns
    -------
    numpy.ndarray
        Coordination numbers for matching sites.
    """

    if crystal_nn is None:
        crystal_nn = create_crystal_nn()

    coordination_numbers: List[float] = []

    for site_index, site in enumerate(structure):

        if get_element_symbol(site) != element_symbol:
            continue

        coordination_numbers.append(
            calculate_site_coordination(
                structure=structure,
                site_index=site_index,
                crystal_nn=crystal_nn,
            )
        )

    return np.asarray(
        coordination_numbers,
        dtype=float,
    )


def calculate_first_shell_gan_bonds(
    structure,
    crystal_nn: CrystalNN | None = None,
) -> np.ndarray:
    """
    Determine unique first-shell Ga-N bond distances.

    CrystalNN determines the local first coordination shell. Only
    heteroatomic Ga-N pairs are retained.

    Duplicate reciprocal bonds are removed.

    Parameters
    ----------
    structure
        Pymatgen Structure.

    crystal_nn : CrystalNN, optional
        Existing CrystalNN analyzer.

    Returns
    -------
    numpy.ndarray
        Unique first-shell Ga-N bond distances in Angstrom.
    """

    if crystal_nn is None:
        crystal_nn = create_crystal_nn()

    bond_distances: List[float] = []
    visited_pairs = set()

    for site_index, site in enumerate(structure):

        site_element = get_element_symbol(site)

        if site_element not in {"Ga", "N"}:
            continue

        neighbor_info = crystal_nn.get_nn_info(
            structure,
            site_index,
        )

        for neighbor in neighbor_info:

            neighbor_site = neighbor["site"]
            neighbor_index = neighbor.get(
                "site_index"
            )

            neighbor_element = get_element_symbol(
                neighbor_site
            )

            if {
                site_element,
                neighbor_element,
            } != {"Ga", "N"}:
                continue

            # When the original site index is available, use it
            # to remove reciprocal duplicates.
            if neighbor_index is not None:

                pair = tuple(
                    sorted(
                        (
                            int(site_index),
                            int(neighbor_index),
                        )
                    )
                )

                if pair in visited_pairs:
                    continue

                visited_pairs.add(pair)

            distance = structure.get_distance(
                site_index,
                int(neighbor_index),
            )

            bond_distances.append(
                float(distance)
            )

    return np.asarray(
        bond_distances,
        dtype=float,
    )


def summarize_coordination(
    structure,
) -> Dict[str, float]:
    """
    Calculate coordination and first-shell bond statistics for GaN.

    Returns
    -------
    dict
        Structural coordination descriptors.
    """

    crystal_nn = create_crystal_nn()

    failed_sites = 0

    ga_coordination = []
    n_coordination = []

    for site_index, site in enumerate(structure):

        element = get_element_symbol(site)

        if element not in {"Ga", "N"}:
            continue

        try:
            coordination = calculate_site_coordination(
                structure=structure,
                site_index=site_index,
                crystal_nn=crystal_nn,
            )

            if element == "Ga":
                ga_coordination.append(
                    coordination
                )

            elif element == "N":
                n_coordination.append(
                    coordination
                )

        except Exception:
            failed_sites += 1

    ga_coordination = np.asarray(
        ga_coordination,
        dtype=float,
    )

    n_coordination = np.asarray(
        n_coordination,
        dtype=float,
    )

    try:
        bond_distances = (
            calculate_first_shell_gan_bonds(
                structure=structure,
                crystal_nn=crystal_nn,
            )
        )

    except Exception:
        bond_distances = np.asarray(
            [],
            dtype=float,
        )

    return {
        "mean_Ga_coordination": (
            float(np.mean(ga_coordination))
            if ga_coordination.size
            else np.nan
        ),

        "std_Ga_coordination": (
            float(np.std(ga_coordination))
            if ga_coordination.size
            else np.nan
        ),

        "mean_N_coordination": (
            float(np.mean(n_coordination))
            if n_coordination.size
            else np.nan
        ),

        "std_N_coordination": (
            float(np.std(n_coordination))
            if n_coordination.size
            else np.nan
        ),

        "number_of_first_shell_GaN_bonds":
            int(bond_distances.size),

        "mean_first_shell_GaN_distance_A": (
            float(np.mean(bond_distances))
            if bond_distances.size
            else np.nan
        ),

        "min_first_shell_GaN_distance_A": (
            float(np.min(bond_distances))
            if bond_distances.size
            else np.nan
        ),

        "max_first_shell_GaN_distance_A": (
            float(np.max(bond_distances))
            if bond_distances.size
            else np.nan
        ),

        "std_first_shell_GaN_distance_A": (
            float(np.std(bond_distances))
            if bond_distances.size
            else np.nan
        ),

        "failed_sites":
            int(failed_sites),
    }


def analyze_coordination_dataframe(
    dataframe: pd.DataFrame,
    structure_column: str = "structure",
    material_id_column: str = "material_id",
) -> pd.DataFrame:
    """
    Perform coordination analysis for multiple GaN structures.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataset containing Pymatgen Structure objects.

    structure_column : str, default="structure"
        Structure column.

    material_id_column : str, default="material_id"
        Material identifier column.

    Returns
    -------
    pandas.DataFrame
        Coordination-analysis table.
    """

    records = []

    for _, row in dataframe.iterrows():

        structure = row[
            structure_column
        ]

        coordination = summarize_coordination(
            structure
        )

        record = {
            "material_id":
                row[material_id_column],

            **coordination,
        }

        records.append(record)

    return pd.DataFrame(records)
