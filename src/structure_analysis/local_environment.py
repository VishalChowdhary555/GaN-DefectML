"""
Local-environment analysis utilities for GaN-DefectML.

This module provides periodic neighbor-search and local structural
descriptors for pristine and defective GaN configurations.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


def periodic_cartesian_distance(
    lattice,
    fractional_a: Sequence[float],
    fractional_b: Sequence[float],
) -> float:
    """
    Calculate the minimum-image Cartesian distance between two
    fractional coordinates.

    Parameters
    ----------
    lattice
        Pymatgen Lattice object.

    fractional_a, fractional_b
        Fractional coordinates.

    Returns
    -------
    float
        Minimum periodic distance in Angstrom.
    """

    fractional_difference = (
        np.asarray(
            fractional_a,
            dtype=float,
        )
        -
        np.asarray(
            fractional_b,
            dtype=float,
        )
    )

    fractional_difference -= np.round(
        fractional_difference
    )

    cartesian_difference = (
        lattice.get_cartesian_coords(
            fractional_difference
        )
    )

    return float(
        np.linalg.norm(
            cartesian_difference
        )
    )


def minimum_distance_to_structure(
    structure,
    fractional_coordinate: Sequence[float],
) -> float:
    """
    Calculate the minimum periodic distance from an arbitrary
    fractional coordinate to any atom in a structure.
    """

    distances = [
        periodic_cartesian_distance(
            lattice=structure.lattice,
            fractional_a=fractional_coordinate,
            fractional_b=site.frac_coords,
        )
        for site in structure
    ]

    return float(
        min(distances)
    )


def get_neighbors_from_coordinate(
    structure,
    center_fractional: Sequence[float],
    radius: float = 4.0,
    minimum_distance: float = 1e-3,
) -> List[Dict]:
    """
    Find atoms around an arbitrary reference coordinate.

    This is useful for vacancies, interstitial sites, and defect
    centers that do not necessarily coincide with a lattice atom.

    Parameters
    ----------
    structure
        Pymatgen Structure.

    center_fractional
        Fractional coordinates of the reference point.

    radius : float, default=4.0
        Neighbor-search radius in Angstrom.

    minimum_distance : float, default=1e-3
        Distances smaller than this are excluded.

    Returns
    -------
    list of dict
        Neighbor index, species, and distance.
    """

    center_cartesian = (
        structure.lattice
        .get_cartesian_coords(
            center_fractional
        )
    )

    neighbors = (
        structure.get_sites_in_sphere(
            center_cartesian,
            r=radius,
            include_index=True,
        )
    )

    records = []

    for neighbor in neighbors:

        site = neighbor[0]
        distance = float(
            neighbor[1]
        )
        site_index = int(
            neighbor[2]
        )

        if distance <= minimum_distance:
            continue

        records.append(
            {
                "site_index":
                    site_index,

                "species":
                    site.specie.symbol,

                "distance_A":
                    distance,
            }
        )

    records.sort(
        key=lambda item:
            item["distance_A"]
    )

    return records


def nearby_species_distances(
    structure,
    center_fractional: Sequence[float],
    target_species: str,
    radius: float = 4.0,
    minimum_distance: float = 1e-3,
) -> List[float]:
    """
    Return periodic distances from a reference coordinate to atoms
    of a selected species.
    """

    neighbors = get_neighbors_from_coordinate(
        structure=structure,
        center_fractional=center_fractional,
        radius=radius,
        minimum_distance=minimum_distance,
    )

    distances = [
        neighbor["distance_A"]
        for neighbor in neighbors
        if neighbor["species"]
        == target_species
    ]

    return sorted(distances)


def calculate_gan_neighbor_distances(
    structure,
    cutoff_radius: float = 3.5,
) -> np.ndarray:
    """
    Calculate unique Ga-N pair distances within a radial cutoff.

    Reciprocal pairs are counted only once.

    Parameters
    ----------
    structure
        Pymatgen Structure.

    cutoff_radius : float, default=3.5
        Maximum Ga-N pair distance.

    Returns
    -------
    numpy.ndarray
        Unique Ga-N pair distances.
    """

    distances = []
    visited_pairs = set()

    for site_index, site in enumerate(
        structure
    ):

        site_element = (
            site.specie.symbol
        )

        if site_element not in {
            "Ga",
            "N",
        }:
            continue

        neighbors = structure.get_neighbors(
            site,
            r=cutoff_radius,
            include_index=True,
        )

        for neighbor in neighbors:

            neighbor_element = (
                neighbor.specie.symbol
            )

            if {
                site_element,
                neighbor_element,
            } != {
                "Ga",
                "N",
            }:
                continue

            neighbor_index = int(
                neighbor.index
            )

            pair = tuple(
                sorted(
                    (
                        site_index,
                        neighbor_index,
                    )
                )
            )

            if pair in visited_pairs:
                continue

            visited_pairs.add(
                pair
            )

            distances.append(
                float(
                    neighbor.nn_distance
                )
            )

    return np.asarray(
        distances,
        dtype=float,
    )


def summarize_gan_neighbor_distances(
    structure,
    cutoff_radius: float = 3.5,
) -> Dict[str, float]:
    """
    Summarize Ga-N neighbor distances within a radial cutoff.
    """

    distances = calculate_gan_neighbor_distances(
        structure=structure,
        cutoff_radius=cutoff_radius,
    )

    return {
        "number_of_GaN_neighbor_pairs":
            int(distances.size),

        "minimum_GaN_distance_A":
            float(np.min(distances))
            if distances.size
            else np.nan,

        "mean_GaN_distance_A":
            float(np.mean(distances))
            if distances.size
            else np.nan,

        "maximum_GaN_distance_A":
            float(np.max(distances))
            if distances.size
            else np.nan,

        "std_GaN_distance_A":
            float(np.std(distances))
            if distances.size
            else np.nan,
    }


def characterize_local_environment(
    structure,
    center_fractional: Sequence[float],
    radius: float = 4.0,
    nearest_neighbor_count: int = 12,
) -> Dict:
    """
    Generate local defect-centered structural descriptors.

    Parameters
    ----------
    structure
        Pymatgen Structure.

    center_fractional
        Defect-center fractional coordinates.

    radius : float, default=4.0
        Search radius.

    nearest_neighbor_count : int, default=12
        Maximum number of nearest atoms retained.

    Returns
    -------
    dict
        Local-environment descriptors.
    """

    neighbors = get_neighbors_from_coordinate(
        structure=structure,
        center_fractional=center_fractional,
        radius=radius,
    )

    nearest_neighbors = neighbors[
        :nearest_neighbor_count
    ]

    nearest_distances = np.asarray(
        [
            neighbor["distance_A"]
            for neighbor in nearest_neighbors
        ],
        dtype=float,
    )

    nearest_species = [
        neighbor["species"]
        for neighbor in nearest_neighbors
    ]

    first_four = (
        nearest_neighbors[:4]
    )

    first_six = (
        nearest_neighbors[:6]
    )

    first_four_distances = np.asarray(
        [
            neighbor["distance_A"]
            for neighbor in first_four
        ],
        dtype=float,
    )

    has_four_neighbors = (
        len(first_four_distances) == 4
    )

    return {
        "neighbors_within_4A":
            len(neighbors),

        "nearest_neighbor_distance_A":
            float(
                nearest_distances[0]
            )
            if nearest_distances.size
            else np.nan,

        "mean_4_nearest_distance_A":
            float(
                np.mean(
                    first_four_distances
                )
            )
            if has_four_neighbors
            else np.nan,

        "std_4_nearest_distance_A":
            float(
                np.std(
                    first_four_distances
                )
            )
            if has_four_neighbors
            else np.nan,

        "min_4_nearest_distance_A":
            float(
                np.min(
                    first_four_distances
                )
            )
            if has_four_neighbors
            else np.nan,

        "max_4_nearest_distance_A":
            float(
                np.max(
                    first_four_distances
                )
            )
            if has_four_neighbors
            else np.nan,

        "range_4_nearest_distance_A":
            float(
                np.max(
                    first_four_distances
                )
                -
                np.min(
                    first_four_distances
                )
            )
            if has_four_neighbors
            else np.nan,

        "Ga_among_4_nearest":
            sum(
                neighbor["species"] == "Ga"
                for neighbor in first_four
            ),

        "N_among_4_nearest":
            sum(
                neighbor["species"] == "N"
                for neighbor in first_four
            ),

        "dopant_among_4_nearest":
            sum(
                neighbor["species"]
                not in {
                    "Ga",
                    "N",
                }
                for neighbor in first_four
            ),

        "Ga_among_6_nearest":
            sum(
                neighbor["species"] == "Ga"
                for neighbor in first_six
            ),

        "N_among_6_nearest":
            sum(
                neighbor["species"] == "N"
                for neighbor in first_six
            ),

        "nearest_species_sequence":
            nearest_species,

        "nearest_distance_sequence_A":
            [
                round(
                    value,
                    6,
                )
                for value
                in nearest_distances
            ],
    }


def analyze_local_environment_dataframe(
    dataframe: pd.DataFrame,
    center_dataframe: pd.DataFrame,
    structure_column: str = "structure_object",
    configuration_id_column: str = "configuration_id",
    radius: float = 4.0,
    nearest_neighbor_count: int = 12,
) -> pd.DataFrame:
    """
    Analyze defect-centered local environments for many structures.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Structure library containing configuration IDs and structures.

    center_dataframe : pandas.DataFrame
        Table containing the defect-center fractional coordinates.

    Returns
    -------
    pandas.DataFrame
        Local-environment feature table.
    """

    records = []

    for _, row in dataframe.iterrows():

        configuration_id = row[
            configuration_id_column
        ]

        center_rows = (
            center_dataframe[
                center_dataframe[
                    configuration_id_column
                ] == configuration_id
            ]
        )

        if len(center_rows) != 1:
            raise ValueError(
                "Expected exactly one defect-center row "
                f"for {configuration_id}, found "
                f"{len(center_rows)}."
            )

        center_row = (
            center_rows.iloc[0]
        )

        center_fractional = np.asarray(
            [
                center_row[
                    "defect_center_fractional_x"
                ],
                center_row[
                    "defect_center_fractional_y"
                ],
                center_row[
                    "defect_center_fractional_z"
                ],
            ],
            dtype=float,
        )

        descriptors = (
            characterize_local_environment(
                structure=row[
                    structure_column
                ],
                center_fractional=
                    center_fractional,
                radius=radius,
                nearest_neighbor_count=
                    nearest_neighbor_count,
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
