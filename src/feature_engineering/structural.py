"""
Structural descriptor utilities for GaN-DefectML.

This module generates global crystallographic descriptors from pristine
and defective structures, including lattice parameters, symmetry,
density, volume, and simple dimensionless lattice ratios.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def analyze_global_structure(
    structure,
    symprec: float = 0.01,
    angle_tolerance: float = 5.0,
) -> Dict[str, float]:
    """
    Generate global structural descriptors for one crystal structure.

    Parameters
    ----------
    structure
        Pymatgen Structure object.

    symprec : float, default=0.01
        Symmetry tolerance used by SpacegroupAnalyzer.

    angle_tolerance : float, default=5.0
        Angular symmetry tolerance in degrees.

    Returns
    -------
    dict
        Global structural descriptor dictionary.
    """

    lattice = structure.lattice

    analyzer = SpacegroupAnalyzer(
        structure,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )

    number_of_sites = len(structure)

    return {
        "number_of_sites":
            int(number_of_sites),

        "lattice_a_A":
            float(lattice.a),

        "lattice_b_A":
            float(lattice.b),

        "lattice_c_A":
            float(lattice.c),

        "lattice_alpha_deg":
            float(lattice.alpha),

        "lattice_beta_deg":
            float(lattice.beta),

        "lattice_gamma_deg":
            float(lattice.gamma),

        "cell_volume_A3":
            float(structure.volume),

        "volume_per_atom_A3":
            (
                float(
                    structure.volume
                    / number_of_sites
                )
                if number_of_sites > 0
                else np.nan
            ),

        "density_g_cm3":
            float(structure.density),

        "space_group_number":
            int(
                analyzer
                .get_space_group_number()
            ),

        "space_group_symbol":
            analyzer
            .get_space_group_symbol(),

        "crystal_system":
            analyzer
            .get_crystal_system(),

        "is_centrosymmetric":
            bool(
                analyzer.is_laue()
            ),

        "lattice_c_over_a":
            (
                float(
                    lattice.c
                    / lattice.a
                )
                if lattice.a != 0
                else np.nan
            ),

        "lattice_b_over_a":
            (
                float(
                    lattice.b
                    / lattice.a
                )
                if lattice.a != 0
                else np.nan
            ),
    }


def build_structural_feature_dataframe(
    structure_dataframe: pd.DataFrame,
    structure_column: str = "structure_object",
    configuration_id_column: str = "configuration_id",
    symprec: float = 0.01,
    angle_tolerance: float = 5.0,
) -> pd.DataFrame:
    """
    Generate global structural descriptors for a structure library.

    Parameters
    ----------
    structure_dataframe : pandas.DataFrame
        Table containing configuration IDs and structures.

    structure_column : str, default="structure_object"
        Column containing Pymatgen Structure objects.

    configuration_id_column : str, default="configuration_id"
        Configuration identifier column.

    Returns
    -------
    pandas.DataFrame
        Structural descriptor table.
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
            analyze_global_structure(
                structure=structure,
                symprec=symprec,
                angle_tolerance=
                    angle_tolerance,
            )
        )

        records.append(
            {
                "configuration_id":
                    configuration_id,

                **descriptors,
            }
        )

    return pd.DataFrame(records)


def identify_constant_structural_features(
    structural_feature_df: pd.DataFrame,
    configuration_id_column: str = "configuration_id",
) -> list[str]:
    """
    Identify structural features that are constant across all samples.

    This is particularly useful for unrelaxed defect libraries where
    lattice parameters and cell volume may remain unchanged.
    """

    candidate_df = (
        structural_feature_df
        .drop(
            columns=[
                configuration_id_column
            ],
            errors="ignore",
        )
    )

    numeric_columns = (
        candidate_df
        .select_dtypes(
            include=[
                np.number,
                "bool",
            ]
        )
        .columns
    )

    constant_columns = [
        column
        for column in numeric_columns
        if candidate_df[column]
        .nunique(dropna=False)
        <= 1
    ]

    return constant_columns


def remove_constant_structural_features(
    structural_feature_df: pd.DataFrame,
    configuration_id_column: str = "configuration_id",
) -> pd.DataFrame:
    """
    Remove constant numerical structural descriptors.

    The configuration identifier is always preserved.
    """

    constant_columns = (
        identify_constant_structural_features(
            structural_feature_df=
                structural_feature_df,
            configuration_id_column=
                configuration_id_column,
        )
    )

    return structural_feature_df.drop(
        columns=constant_columns,
        errors="ignore",
    )


def validate_structural_features(
    structural_feature_df: pd.DataFrame,
) -> bool:
    """
    Validate the numerical structural descriptor block.

    Raises
    ------
    ValueError
        If physically invalid structural quantities are found.
    """

    positive_columns = [
        "number_of_sites",
        "lattice_a_A",
        "lattice_b_A",
        "lattice_c_A",
        "cell_volume_A3",
        "volume_per_atom_A3",
        "density_g_cm3",
    ]

    for column in positive_columns:

        if column not in (
            structural_feature_df.columns
        ):
            continue

        numeric_values = pd.to_numeric(
            structural_feature_df[
                column
            ],
            errors="coerce",
        )

        invalid_mask = (
            numeric_values.isna()
            |
            (numeric_values <= 0)
        )

        if invalid_mask.any():

            if (
                "configuration_id"
                in structural_feature_df.columns
            ):

                invalid_configurations = (
                    structural_feature_df.loc[
                        invalid_mask,
                        "configuration_id",
                    ]
                    .tolist()
                )

                raise ValueError(
                    f"Invalid values found in "
                    f"{column} for configurations: "
                    f"{invalid_configurations}"
                )

            raise ValueError(
                f"Invalid values found in "
                f"{column}."
            )

    return True
