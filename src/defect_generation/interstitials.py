"""
Interstitial-generation utilities for GaN-DefectML.

This module provides helpers for:
- searching periodic geometric voids,
- ranking candidate interstitial sites,
- constructing local-environment fingerprints,
- reducing candidates into distinct geometric families,
- and generating Ga and N interstitial structures.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


def periodic_fractional_distance(
    lattice,
    fractional_a: Sequence[float],
    fractional_b: Sequence[float],
) -> float:
    """
    Return minimum-image Cartesian distance between two fractional
    coordinates.
    """

    distance, _ = lattice.get_distance_and_image(
        fractional_a,
        fractional_b,
    )

    return float(distance)


def minimum_host_distance(
    structure,
    fractional_coordinate: Sequence[float],
) -> float:
    """
    Return the minimum periodic distance from a candidate point
    to any host atom.
    """

    distances = [
        periodic_fractional_distance(
            lattice=structure.lattice,
            fractional_a=fractional_coordinate,
            fractional_b=site.frac_coords,
        )
        for site in structure
    ]

    return float(min(distances))


def generate_fractional_grid(
    grid_points_per_axis: int = 10,
) -> np.ndarray:
    """
    Generate a regular fractional-coordinate grid in [0, 1).

    Parameters
    ----------
    grid_points_per_axis : int, default=10
        Number of grid samples along each fractional axis.

    Returns
    -------
    numpy.ndarray
        Array of shape (N, 3).
    """

    if grid_points_per_axis < 2:
        raise ValueError(
            "grid_points_per_axis must be at least 2."
        )

    coordinates = np.linspace(
        0.0,
        1.0,
        grid_points_per_axis,
        endpoint=False,
    )

    mesh = np.array(
        np.meshgrid(
            coordinates,
            coordinates,
            coordinates,
            indexing="ij",
        )
    )

    return (
        mesh.reshape(3, -1).T
    )


def find_void_candidates(
    structure,
    grid_points_per_axis: int = 10,
    minimum_allowed_distance_A: float = 1.5,
) -> pd.DataFrame:
    """
    Search a periodic fractional grid for empty interstitial regions.

    Candidate points are retained only when their minimum host-atom
    distance exceeds the selected threshold.

    Returns
    -------
    pandas.DataFrame
        Candidate fractional coordinates and minimum host distance.
    """

    grid = generate_fractional_grid(
        grid_points_per_axis=
            grid_points_per_axis
    )

    records = []

    for fractional_coordinate in grid:

        distance = minimum_host_distance(
            structure=structure,
            fractional_coordinate=
                fractional_coordinate,
        )

        if distance < minimum_allowed_distance_A:
            continue

        records.append(
            {
                "fractional_x":
                    float(
                        fractional_coordinate[0]
                    ),

                "fractional_y":
                    float(
                        fractional_coordinate[1]
                    ),

                "fractional_z":
                    float(
                        fractional_coordinate[2]
                    ),

                "minimum_host_distance_A":
                    distance,
            }
        )

    candidate_df = pd.DataFrame(
        records
    )

    if candidate_df.empty:
        return candidate_df

    candidate_df = (
        candidate_df
        .sort_values(
            "minimum_host_distance_A",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    candidate_df.insert(
        0,
        "candidate_id",
        [
            f"I_site_{index + 1}"
            for index in range(
                len(candidate_df)
            )
        ],
    )

    return candidate_df


def periodic_candidate_distance(
    lattice,
    candidate_a: Sequence[float],
    candidate_b: Sequence[float],
) -> float:
    """
    Calculate periodic distance between two candidate coordinates.
    """

    return periodic_fractional_distance(
        lattice=lattice,
        fractional_a=candidate_a,
        fractional_b=candidate_b,
    )


def reduce_near_duplicate_candidates(
    structure,
    candidate_df: pd.DataFrame,
    minimum_separation_A: float = 0.5,
    maximum_candidates: int | None = None,
) -> pd.DataFrame:
    """
    Remove geometrically redundant candidate voids.

    Candidates are processed from largest to smallest void radius.
    A point is retained only when it is sufficiently separated from
    all previously accepted points.
    """

    if candidate_df.empty:
        return candidate_df.copy()

    retained_rows = []

    for _, row in candidate_df.iterrows():

        coordinate = np.asarray(
            [
                row["fractional_x"],
                row["fractional_y"],
                row["fractional_z"],
            ],
            dtype=float,
        )

        is_distinct = True

        for retained_row in retained_rows:

            retained_coordinate = np.asarray(
                [
                    retained_row["fractional_x"],
                    retained_row["fractional_y"],
                    retained_row["fractional_z"],
                ],
                dtype=float,
            )

            distance = periodic_candidate_distance(
                lattice=structure.lattice,
                candidate_a=coordinate,
                candidate_b=retained_coordinate,
            )

            if distance < minimum_separation_A:
                is_distinct = False
                break

        if is_distinct:
            retained_rows.append(
                row.to_dict()
            )

        if (
            maximum_candidates is not None
            and len(retained_rows)
            >= maximum_candidates
        ):
            break

    return pd.DataFrame(
        retained_rows
    ).reset_index(drop=True)


def get_local_species_distances(
    structure,
    fractional_coordinate: Sequence[float],
    cutoff_radius_A: float = 3.5,
) -> Dict[str, List[float]]:
    """
    Collect neighboring Ga and N distances around an arbitrary
    interstitial coordinate.
    """

    center_cartesian = (
        structure.lattice
        .get_cartesian_coords(
            fractional_coordinate
        )
    )

    neighbors = structure.get_sites_in_sphere(
        center_cartesian,
        r=cutoff_radius_A,
        include_index=True,
    )

    ga_distances = []
    n_distances = []

    for neighbor in neighbors:

        site = neighbor[0]
        distance = float(
            neighbor[1]
        )

        if distance <= 1e-6:
            continue

        symbol = site.specie.symbol

        if symbol == "Ga":
            ga_distances.append(distance)

        elif symbol == "N":
            n_distances.append(distance)

    return {
        "Ga":
            sorted(ga_distances),

        "N":
            sorted(n_distances),
    }


def build_environment_fingerprint(
    structure,
    fractional_coordinate: Sequence[float],
    neighbor_count: int = 4,
    cutoff_radius_A: float = 3.5,
    distance_decimals: int = 3,
) -> tuple:
    """
    Build a local chemical-environment fingerprint.

    The fingerprint records the species and rounded distance of the
    nearest neighboring atoms around an interstitial candidate.

    Returns
    -------
    tuple
        Ordered ``(species, distance)`` pairs.
    """

    center_cartesian = (
        structure.lattice
        .get_cartesian_coords(
            fractional_coordinate
        )
    )

    neighbors = structure.get_sites_in_sphere(
        center_cartesian,
        r=cutoff_radius_A,
        include_index=True,
    )

    local_environment = []

    for neighbor in neighbors:

        site = neighbor[0]
        distance = float(
            neighbor[1]
        )

        if distance <= 1e-6:
            continue

        local_environment.append(
            (
                site.specie.symbol,
                distance,
            )
        )

    local_environment.sort(
        key=lambda item: item[1]
    )

    local_environment = local_environment[
        :neighbor_count
    ]

    return tuple(
        (
            species,
            round(
                distance,
                distance_decimals,
            ),
        )
        for species, distance
        in local_environment
    )


def add_environment_fingerprints(
    structure,
    candidate_df: pd.DataFrame,
    neighbor_count: int = 4,
    cutoff_radius_A: float = 3.5,
) -> pd.DataFrame:
    """
    Add local chemical-environment fingerprints to candidate sites.
    """

    output = candidate_df.copy()

    fingerprints = []

    for _, row in output.iterrows():

        coordinate = np.asarray(
            [
                row["fractional_x"],
                row["fractional_y"],
                row["fractional_z"],
            ],
            dtype=float,
        )

        fingerprint = build_environment_fingerprint(
            structure=structure,
            fractional_coordinate=coordinate,
            neighbor_count=neighbor_count,
            cutoff_radius_A=cutoff_radius_A,
        )

        fingerprints.append(
            fingerprint
        )

    output[
        "environment_fingerprint"
    ] = fingerprints

    return output


def group_environment_families(
    candidate_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign geometric environment-family labels using fingerprints.

    Equivalent fingerprints are assigned to the same family.
    """

    if (
        "environment_fingerprint"
        not in candidate_df.columns
    ):
        raise ValueError(
            "Candidate DataFrame must contain "
            "'environment_fingerprint'."
        )

    output = candidate_df.copy()

    fingerprint_to_family = {}

    family_ids = []

    next_family_index = 1

    for fingerprint in output[
        "environment_fingerprint"
    ]:

        if fingerprint not in fingerprint_to_family:

            fingerprint_to_family[
                fingerprint
            ] = (
                f"Geo_I_"
                f"{next_family_index}"
            )

            next_family_index += 1

        family_ids.append(
            fingerprint_to_family[
                fingerprint
            ]
        )

    output[
        "environment_family"
    ] = family_ids

    return output


def select_environment_representatives(
    candidate_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select one candidate per local environment family.

    The candidate with the largest minimum host distance is retained.
    """

    if "environment_family" not in candidate_df.columns:
        raise ValueError(
            "Candidate DataFrame must contain "
            "'environment_family'."
        )

    representatives = (
        candidate_df
        .sort_values(
            "minimum_host_distance_A",
            ascending=False,
        )
        .drop_duplicates(
            subset=["environment_family"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    return representatives


def create_interstitial(
    pristine_structure,
    fractional_coordinate: Sequence[float],
    species: str,
):
    """
    Insert an interstitial atom into a copy of the pristine host.
    """

    interstitial_structure = (
        pristine_structure.copy()
    )

    interstitial_structure.append(
        species,
        fractional_coordinate,
        coords_are_cartesian=False,
        validate_proximity=False,
    )

    metadata = {
        "interstitial_species":
            species,

        "defect_species":
            species,

        "defect_center_fractional":
            np.asarray(
                fractional_coordinate,
                dtype=float,
            ).copy(),

        "defect_atom_index":
            int(
                len(
                    interstitial_structure
                )
                - 1
            ),

        "original_number_of_sites":
            int(
                len(
                    pristine_structure
                )
            ),

        "defective_number_of_sites":
            int(
                len(
                    interstitial_structure
                )
            ),
    }

    return (
        interstitial_structure,
        metadata,
    )


def generate_interstitial_library(
    pristine_structure,
    representative_site_df: pd.DataFrame,
    species_list: Iterable[str] = ("Ga", "N"),
) -> dict:
    """
    Generate interstitial configurations for every retained
    geometric environment.

    Parameters
    ----------
    pristine_structure
        Pristine GaN supercell.

    representative_site_df
        Must contain ``environment_family`` and fractional-coordinate
        columns.

    species_list
        Species inserted at each retained site.

    Returns
    -------
    dict
        Configuration ID -> structure and metadata.
    """

    generated = {}

    for _, row in representative_site_df.iterrows():

        environment_family = row[
            "environment_family"
        ]

        fractional_coordinate = np.asarray(
            [
                row["fractional_x"],
                row["fractional_y"],
                row["fractional_z"],
            ],
            dtype=float,
        )

        for species in species_list:

            configuration_id = (
                f"{species}_i_"
                f"{environment_family}"
            )

            structure, metadata = (
                create_interstitial(
                    pristine_structure=
                        pristine_structure,
                    fractional_coordinate=
                        fractional_coordinate,
                    species=species,
                )
            )

            metadata[
                "configuration_id"
            ] = configuration_id

            metadata[
                "defect_type"
            ] = "interstitial"

            metadata[
                "environment_family"
            ] = environment_family

            metadata[
                "geometric_site_id"
            ] = row.get(
                "candidate_id"
            )

            metadata[
                "minimum_host_distance_A"
            ] = row.get(
                "minimum_host_distance_A"
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


def validate_interstitial_structure(
    pristine_structure,
    interstitial_structure,
    inserted_species: str,
) -> bool:
    """
    Validate an interstitial configuration.

    Expected behavior:
    - total atom count increases by one,
    - inserted species count increases by one.
    """

    if len(interstitial_structure) != (
        len(pristine_structure) + 1
    ):
        raise ValueError(
            "Interstitial structure must contain exactly "
            "one additional atom."
        )

    pristine_count = sum(
        site.specie.symbol
        == inserted_species
        for site in pristine_structure
    )

    interstitial_count = sum(
        site.specie.symbol
        == inserted_species
        for site in interstitial_structure
    )

    if interstitial_count != (
        pristine_count + 1
    ):
        raise ValueError(
            f"Expected one additional {inserted_species} atom."
        )

    return True
