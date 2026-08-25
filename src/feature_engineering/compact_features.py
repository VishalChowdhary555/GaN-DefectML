"""
Compact physics-informed feature selection for GaN-DefectML.

This module reduces the full structural-chemical descriptor table into
a smaller, interpretable feature representation suitable for conventional
machine learning and explainable-AI analysis.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


DEFAULT_COMPACT_FEATURE_GROUPS = {
    "defect_identity": [
        "is_pristine",
        "is_vacancy",
        "is_antisite",
        "is_substitutional",
        "is_interstitial",
        "chemical_operation_none",
        "chemical_operation_removal",
        "chemical_operation_replacement",
        "chemical_operation_insertion",
    ],

    "composition": [
        "number_of_elements",
        "total_atoms_composition",
        "fraction_Ga",
        "fraction_N",
        "fraction_Mg",
        "fraction_Si",
        "fraction_O",
        "fraction_C",
        "mean_atomic_number",
        "std_atomic_number",
        "mean_atomic_mass",
        "std_atomic_mass",
        "mean_electronegativity",
        "std_electronegativity",
        "mean_atomic_radius_A",
        "std_atomic_radius_A",
    ],

    "global_structure": [
        "number_of_sites",
        "volume_per_atom_A3",
        "density_g_cm3",
        "space_group_number",
        "is_centrosymmetric",
    ],

    "local_geometry": [
        "neighbors_within_4A",
        "nearest_neighbor_distance_A",
        "mean_4_nearest_distance_A",
        "std_4_nearest_distance_A",
        "range_4_nearest_distance_A",
        "Ga_among_4_nearest",
        "N_among_4_nearest",
        "dopant_among_4_nearest",
        "Ga_among_6_nearest",
        "N_among_6_nearest",
    ],

    "neighbor_shell": [
        "neighbor_1_distance_A",
        "neighbor_2_distance_A",
        "neighbor_3_distance_A",
        "neighbor_4_distance_A",
        "neighbor_5_distance_A",
        "neighbor_6_distance_A",
        "neighbor_8_distance_A",
        "neighbor_10_distance_A",
        "neighbor_12_distance_A",
    ],

    "signed_chemical_change": [
        "delta_atomic_number",
        "delta_atomic_mass",
        "delta_electronegativity",
        "delta_atomic_radius_A",
        "delta_group",
        "delta_row",
        "delta_first_ionization_energy_eV",
        "delta_electron_affinity_eV",
        "delta_valence_electrons",
    ],

    "normalized_chemical_change": [
        "normalized_delta_atomic_number",
        "normalized_delta_atomic_mass",
        "normalized_delta_electronegativity",
        "normalized_delta_atomic_radius_A",
        "normalized_delta_group",
        "normalized_delta_first_ionization_energy_eV",
        "normalized_delta_electron_affinity_eV",
        "normalized_delta_valence_electrons",
    ],

    "summary_mismatch": [
        "combined_chemical_perturbation",
    ],
}


def flatten_feature_groups(
    feature_groups: Dict[str, Iterable[str]],
) -> list[str]:
    """
    Flatten a dictionary of feature groups into one ordered list.
    """

    ordered_features = []

    for group_features in feature_groups.values():
        ordered_features.extend(
            list(group_features)
        )

    return ordered_features


def select_available_compact_features(
    dataframe: pd.DataFrame,
    feature_groups: Optional[
        Dict[str, Iterable[str]]
    ] = None,
) -> tuple[list[str], list[str]]:
    """
    Determine which requested compact features exist in a DataFrame.

    Returns
    -------
    tuple
        ``(available_features, missing_features)``
    """

    if feature_groups is None:
        feature_groups = (
            DEFAULT_COMPACT_FEATURE_GROUPS
        )

    requested_features = (
        flatten_feature_groups(
            feature_groups
        )
    )

    available_features = [
        feature
        for feature in requested_features
        if feature in dataframe.columns
    ]

    missing_features = sorted(
        set(requested_features)
        - set(available_features)
    )

    return (
        available_features,
        missing_features,
    )


def build_initial_compact_matrix(
    dataframe: pd.DataFrame,
    configuration_id_column: str = "configuration_id",
    feature_groups: Optional[
        Dict[str, Iterable[str]]
    ] = None,
) -> pd.DataFrame:
    """
    Build the initial compact physics-informed feature matrix.
    """

    available_features, _ = (
        select_available_compact_features(
            dataframe=dataframe,
            feature_groups=feature_groups,
        )
    )

    output = dataframe[
        [
            configuration_id_column,
            *available_features,
        ]
    ].copy()

    boolean_columns = (
        output
        .select_dtypes(
            include=["bool"]
        )
        .columns
    )

    if len(boolean_columns) > 0:
        output[
            boolean_columns
        ] = output[
            boolean_columns
        ].astype(int)

    return output


def identify_constant_features(
    dataframe: pd.DataFrame,
    configuration_id_column: str = "configuration_id",
) -> list[str]:
    """
    Identify constant numerical features.
    """

    numeric_df = (
        dataframe
        .drop(
            columns=[
                configuration_id_column
            ],
            errors="ignore",
        )
        .select_dtypes(
            include=[
                np.number,
                "bool",
            ]
        )
    )

    return [
        column
        for column in numeric_df.columns
        if numeric_df[column]
        .nunique(dropna=False)
        <= 1
    ]


def identify_duplicate_features(
    dataframe: pd.DataFrame,
    configuration_id_column: str = "configuration_id",
) -> list[str]:
    """
    Identify exact duplicate feature columns.

    The earlier feature in column order is retained.
    """

    feature_df = dataframe.drop(
        columns=[
            configuration_id_column
        ],
        errors="ignore",
    )

    duplicate_columns = []

    column_names = feature_df.columns.tolist()

    for column_index, column_name in enumerate(
        column_names
    ):

        for previous_column in (
            column_names[
                :column_index
            ]
        ):

            if feature_df[
                column_name
            ].equals(
                feature_df[
                    previous_column
                ]
            ):

                duplicate_columns.append(
                    column_name
                )

                break

    return duplicate_columns


def remove_constant_and_duplicate_features(
    dataframe: pd.DataFrame,
    configuration_id_column: str = "configuration_id",
) -> tuple[
    pd.DataFrame,
    list[str],
    list[str],
]:
    """
    Remove constant and exact-duplicate feature columns.

    Returns
    -------
    tuple
        cleaned dataframe,
        removed constant columns,
        removed duplicate columns.
    """

    output = dataframe.copy()

    constant_columns = (
        identify_constant_features(
            dataframe=output,
            configuration_id_column=
                configuration_id_column,
        )
    )

    output = output.drop(
        columns=constant_columns,
        errors="ignore",
    )

    duplicate_columns = (
        identify_duplicate_features(
            dataframe=output,
            configuration_id_column=
                configuration_id_column,
        )
    )

    output = output.drop(
        columns=duplicate_columns,
        errors="ignore",
    )

    return (
        output,
        constant_columns,
        duplicate_columns,
    )


def calculate_absolute_correlation_matrix(
    dataframe: pd.DataFrame,
    configuration_id_column: str = "configuration_id",
) -> pd.DataFrame:
    """
    Calculate the absolute Pearson-correlation matrix.
    """

    numeric_df = (
        dataframe
        .drop(
            columns=[
                configuration_id_column
            ],
            errors="ignore",
        )
        .select_dtypes(
            include=[
                np.number,
                "bool",
            ]
        )
    )

    return numeric_df.corr().abs()


def find_highly_correlated_pairs(
    dataframe: pd.DataFrame,
    correlation_threshold: float = 0.95,
    configuration_id_column: str = "configuration_id",
) -> pd.DataFrame:
    """
    Identify feature pairs above a selected absolute correlation.
    """

    correlation_matrix = (
        calculate_absolute_correlation_matrix(
            dataframe=dataframe,
            configuration_id_column=
                configuration_id_column,
        )
    )

    upper_triangle = (
        correlation_matrix.where(
            np.triu(
                np.ones(
                    correlation_matrix.shape,
                    dtype=bool,
                ),
                k=1,
            )
        )
    )

    records = []

    for column in upper_triangle.columns:

        matching_rows = (
            upper_triangle.index[
                upper_triangle[
                    column
                ]
                >= correlation_threshold
            ]
        )

        for correlated_feature in (
            matching_rows
        ):

            records.append(
                {
                    "feature_1":
                        correlated_feature,

                    "feature_2":
                        column,

                    "absolute_correlation":
                        float(
                            upper_triangle.loc[
                                correlated_feature,
                                column,
                            ]
                        ),
                }
            )

    if not records:
        return pd.DataFrame(
            columns=[
                "feature_1",
                "feature_2",
                "absolute_correlation",
            ]
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            "absolute_correlation",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def remove_highly_correlated_features(
    dataframe: pd.DataFrame,
    correlation_threshold: float = 0.95,
    configuration_id_column: str = "configuration_id",
) -> tuple[
    pd.DataFrame,
    list[str],
]:
    """
    Remove highly correlated redundant features.

    The function preserves the earlier column in the intentionally
    ordered feature list and removes later correlated columns.

    Notes
    -----
    With very small datasets, sample correlations are unstable.
    This should therefore be treated as pragmatic redundancy reduction,
    not as evidence that removed variables are physically unimportant.
    """

    feature_df = dataframe.drop(
        columns=[
            configuration_id_column
        ],
        errors="ignore",
    ).copy()

    numeric_df = (
        feature_df
        .select_dtypes(
            include=[
                np.number,
                "bool",
            ]
        )
    )

    correlation_matrix = (
        numeric_df.corr().abs()
    )

    upper_triangle = (
        correlation_matrix.where(
            np.triu(
                np.ones(
                    correlation_matrix.shape,
                    dtype=bool,
                ),
                k=1,
            )
        )
    )

    columns_to_remove = []

    for column in upper_triangle.columns:

        if (
            upper_triangle[
                column
            ]
            >= correlation_threshold
        ).any():

            columns_to_remove.append(
                column
            )

    reduced_feature_df = (
        feature_df.drop(
            columns=columns_to_remove,
            errors="ignore",
        )
    )

    if (
        configuration_id_column
        in dataframe.columns
    ):

        reduced_dataframe = pd.concat(
            [
                dataframe[
                    [
                        configuration_id_column
                    ]
                ].reset_index(
                    drop=True
                ),

                reduced_feature_df.reset_index(
                    drop=True
                ),
            ],
            axis=1,
        )

    else:
        reduced_dataframe = (
            reduced_feature_df
        )

    return (
        reduced_dataframe,
        columns_to_remove,
    )


def build_compact_physics_informed_matrix(
    dataframe: pd.DataFrame,
    correlation_threshold: float = 0.95,
    configuration_id_column: str = "configuration_id",
    feature_groups: Optional[
        Dict[str, Iterable[str]]
    ] = None,
) -> tuple[
    pd.DataFrame,
    dict,
]:
    """
    Run the complete compact feature-selection pipeline.

    Workflow
    --------
    1. select physics-informed features,
    2. remove constant columns,
    3. remove exact duplicates,
    4. remove highly correlated later features.

    Returns
    -------
    tuple
        final compact DataFrame and selection metadata.
    """

    initial_matrix = (
        build_initial_compact_matrix(
            dataframe=dataframe,
            configuration_id_column=
                configuration_id_column,
            feature_groups=
                feature_groups,
        )
    )

    (
        cleaned_matrix,
        constant_columns,
        duplicate_columns,
    ) = (
        remove_constant_and_duplicate_features(
            dataframe=
                initial_matrix,
            configuration_id_column=
                configuration_id_column,
        )
    )

    (
        final_matrix,
        correlated_columns,
    ) = (
        remove_highly_correlated_features(
            dataframe=
                cleaned_matrix,
            correlation_threshold=
                correlation_threshold,
            configuration_id_column=
                configuration_id_column,
        )
    )

    metadata = {
        "initial_feature_count":
            initial_matrix.shape[1]
            - int(
                configuration_id_column
                in initial_matrix.columns
            ),

        "constant_features_removed":
            constant_columns,

        "duplicate_features_removed":
            duplicate_columns,

        "correlated_features_removed":
            correlated_columns,

        "final_feature_count":
            final_matrix.shape[1]
            - int(
                configuration_id_column
                in final_matrix.columns
            ),

        "correlation_threshold":
            correlation_threshold,
    }

    return (
        final_matrix,
        metadata,
    )


def validate_compact_matrix(
    compact_dataframe: pd.DataFrame,
    configuration_id_column: str = "configuration_id",
) -> bool:
    """
    Validate the compact descriptor matrix.

    Checks:
    - unique configuration IDs,
    - numerical predictor columns,
    - no infinite values.
    """

    if (
        configuration_id_column
        in compact_dataframe.columns
    ):

        if compact_dataframe[
            configuration_id_column
        ].duplicated().any():

            raise ValueError(
                "Duplicate configuration IDs found "
                "in compact descriptor matrix."
            )

    numeric_df = (
        compact_dataframe
        .drop(
            columns=[
                configuration_id_column
            ],
            errors="ignore",
        )
        .select_dtypes(
            include=[
                np.number,
                "bool",
            ]
        )
    )

    if numeric_df.shape[1] != (
        compact_dataframe.shape[1]
        - int(
            configuration_id_column
            in compact_dataframe.columns
        )
    ):

        raise ValueError(
            "Compact descriptor matrix contains "
            "non-numerical predictor columns."
        )

    if np.isinf(
        numeric_df.to_numpy(
            dtype=float
        )
    ).any():

        raise ValueError(
            "Infinite values detected in compact "
            "descriptor matrix."
        )

    return True
