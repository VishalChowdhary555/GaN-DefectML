"""
Validation utilities for GaN-DefectML.

This module provides project-wide consistency checks for:
- configuration identifiers,
- structure libraries,
- feature matrices,
- target tables,
- graph datasets,
- numerical finite-value validation,
- and cross-table alignment.

These functions are intended to fail loudly when the research pipeline
contains inconsistent or incomplete data.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch


# =====================================================================
# Configuration identifiers
# =====================================================================

def validate_unique_configuration_ids(
    dataframe: pd.DataFrame,
    configuration_column: str = "configuration_id",
) -> bool:
    """
    Ensure that configuration identifiers are present and unique.
    """

    if configuration_column not in dataframe.columns:
        raise ValueError(
            f"Missing required column: "
            f"'{configuration_column}'."
        )

    if dataframe[
        configuration_column
    ].isna().any():

        raise ValueError(
            "Configuration identifiers contain missing values."
        )

    duplicated = (
        dataframe[
            configuration_column
        ]
        .duplicated()
    )

    if duplicated.any():

        duplicate_ids = (
            dataframe.loc[
                duplicated,
                configuration_column,
            ]
            .tolist()
        )

        raise ValueError(
            "Duplicate configuration IDs detected: "
            f"{duplicate_ids}"
        )

    return True


def validate_expected_configurations(
    actual_ids: Iterable[str],
    expected_ids: Iterable[str],
) -> dict:
    """
    Compare actual configuration IDs with an expected set.
    """

    actual_ids = set(
        actual_ids
    )

    expected_ids = set(
        expected_ids
    )

    missing = (
        expected_ids
        - actual_ids
    )

    unexpected = (
        actual_ids
        - expected_ids
    )

    passed = bool(
        not missing
        and not unexpected
    )

    return {
        "expected_count":
            len(
                expected_ids
            ),

        "actual_count":
            len(
                actual_ids
            ),

        "missing_configurations":
            missing,

        "unexpected_configurations":
            unexpected,

        "passed":
            passed,
    }


# =====================================================================
# Numerical validation
# =====================================================================

def validate_finite_dataframe(
    dataframe: pd.DataFrame,
    exclude_columns: Sequence[str] = (),
) -> bool:
    """
    Ensure all numerical columns contain finite values.
    """

    numeric_df = (
        dataframe
        .drop(
            columns=list(
                exclude_columns
            ),
            errors="ignore",
        )
        .select_dtypes(
            include=[
                np.number,
                "bool",
            ]
        )
    )

    if numeric_df.empty:
        return True

    values = numeric_df.to_numpy(
        dtype=float
    )

    if np.isnan(values).any():

        nan_columns = (
            numeric_df.columns[
                numeric_df
                .isna()
                .any()
            ]
            .tolist()
        )

        raise ValueError(
            "NaN values detected in columns: "
            f"{nan_columns}"
        )

    if np.isinf(values).any():

        infinite_columns = []

        for column in numeric_df.columns:

            column_values = (
                pd.to_numeric(
                    numeric_df[
                        column
                    ],
                    errors="coerce",
                )
                .to_numpy(
                    dtype=float
                )
            )

            if np.isinf(
                column_values
            ).any():

                infinite_columns.append(
                    column
                )

        raise ValueError(
            "Infinite values detected in columns: "
            f"{infinite_columns}"
        )

    return True


def validate_no_constant_features(
    dataframe: pd.DataFrame,
    exclude_columns: Sequence[str] = (
        "configuration_id",
    ),
) -> bool:
    """
    Ensure no constant predictor columns remain.
    """

    feature_df = dataframe.drop(
        columns=list(
            exclude_columns
        ),
        errors="ignore",
    )

    constant_columns = [
        column
        for column in feature_df.columns
        if feature_df[
            column
        ].nunique(
            dropna=False
        ) <= 1
    ]

    if constant_columns:

        raise ValueError(
            "Constant feature columns remain: "
            f"{constant_columns}"
        )

    return True


# =====================================================================
# Structure-library validation
# =====================================================================

def validate_structure_library(
    structure_dataframe: pd.DataFrame,
    configuration_column: str = "configuration_id",
    structure_column: str = "structure_object",
) -> bool:
    """
    Validate a structure library DataFrame.
    """

    validate_unique_configuration_ids(
        structure_dataframe,
        configuration_column=
            configuration_column,
    )

    if structure_column not in (
        structure_dataframe.columns
    ):

        raise ValueError(
            f"Structure table must contain "
            f"'{structure_column}'."
        )

    empty_structure_ids = []

    for _, row in (
        structure_dataframe.iterrows()
    ):

        structure = row[
            structure_column
        ]

        configuration_id = row[
            configuration_column
        ]

        if structure is None:

            empty_structure_ids.append(
                configuration_id
            )

            continue

        try:
            number_of_sites = len(
                structure
            )

        except Exception as error:

            raise TypeError(
                f"Invalid structure object for "
                f"{configuration_id}."
            ) from error

        if number_of_sites <= 0:

            empty_structure_ids.append(
                configuration_id
            )

    if empty_structure_ids:

        raise ValueError(
            "Empty or missing structures detected for: "
            f"{empty_structure_ids}"
        )

    return True


def validate_structure_composition(
    structure,
    allowed_elements: Iterable[str] = (
        "Ga",
        "N",
        "Mg",
        "Si",
        "O",
        "C",
    ),
) -> bool:
    """
    Ensure that a structure contains only supported project elements.
    """

    allowed_elements = set(
        allowed_elements
    )

    present_elements = {
        site.specie.symbol
        for site in structure
    }

    unsupported = (
        present_elements
        - allowed_elements
    )

    if unsupported:

        raise ValueError(
            "Unsupported elements detected: "
            f"{sorted(unsupported)}"
        )

    return True


# =====================================================================
# Feature-matrix validation
# =====================================================================

def validate_feature_matrix(
    feature_dataframe: pd.DataFrame,
    configuration_column: str = "configuration_id",
    require_finite: bool = True,
    require_nonconstant: bool = True,
) -> bool:
    """
    Validate a supervised feature matrix.
    """

    validate_unique_configuration_ids(
        feature_dataframe,
        configuration_column=
            configuration_column,
    )

    predictor_df = (
        feature_dataframe.drop(
            columns=[
                configuration_column
            ],
            errors="ignore",
        )
    )

    if predictor_df.empty:

        raise ValueError(
            "Feature matrix contains no predictors."
        )

    non_numeric = (
        predictor_df
        .select_dtypes(
            exclude=[
                np.number,
                "bool",
            ]
        )
        .columns
        .tolist()
    )

    if non_numeric:

        raise ValueError(
            "Non-numerical feature columns detected: "
            f"{non_numeric}"
        )

    if require_finite:

        validate_finite_dataframe(
            feature_dataframe,
            exclude_columns=[
                configuration_column
            ],
        )

    if require_nonconstant:

        validate_no_constant_features(
            feature_dataframe,
            exclude_columns=[
                configuration_column
            ],
        )

    return True


# =====================================================================
# Target-table validation
# =====================================================================

def validate_target_alignment(
    feature_dataframe: pd.DataFrame,
    target_dataframe: pd.DataFrame,
    configuration_column: str = "configuration_id",
) -> dict:
    """
    Compare configuration coverage between feature and target tables.
    """

    feature_ids = set(
        feature_dataframe[
            configuration_column
        ]
    )

    target_ids = set(
        target_dataframe[
            configuration_column
        ]
    )

    missing_targets = (
        feature_ids
        - target_ids
    )

    orphan_targets = (
        target_ids
        - feature_ids
    )

    return {
        "feature_configuration_count":
            len(
                feature_ids
            ),

        "target_configuration_count":
            len(
                target_ids
            ),

        "missing_target_configurations":
            missing_targets,

        "orphan_target_configurations":
            orphan_targets,

        "fully_aligned":
            bool(
                not missing_targets
                and not orphan_targets
            ),
    }


# =====================================================================
# Graph validation
# =====================================================================

def validate_graph_tensor_finiteness(
    graph,
) -> bool:
    """
    Ensure graph input tensors contain finite values.
    """

    tensor_attributes = [
        "x",
        "edge_attr",
        "pos",
        "graph_attr",
    ]

    for attribute in tensor_attributes:

        if not hasattr(
            graph,
            attribute,
        ):

            raise AttributeError(
                f"Graph is missing '{attribute}'."
            )

        tensor = getattr(
            graph,
            attribute,
        )

        if not torch.isfinite(
            tensor
        ).all():

            raise ValueError(
                f"Non-finite values detected "
                f"in graph.{attribute} for "
                f"{getattr(graph, 'configuration_id', 'unknown')}."
            )

    return True


def validate_graph_structure_alignment(
    graphs,
    structure_dataframe: pd.DataFrame,
    configuration_column: str = "configuration_id",
    structure_column: str = "structure_object",
) -> bool:
    """
    Ensure graph node counts match corresponding crystal structures.
    """

    structure_lookup = {
        row[
            configuration_column
        ]:
            row[
                structure_column
            ]

        for _, row in (
            structure_dataframe.iterrows()
        )
    }

    for graph in graphs:

        configuration_id = getattr(
            graph,
            "configuration_id",
            None,
        )

        if configuration_id not in (
            structure_lookup
        ):

            raise ValueError(
                f"No corresponding structure found "
                f"for graph '{configuration_id}'."
            )

        structure = structure_lookup[
            configuration_id
        ]

        if int(
            graph.num_nodes
        ) != len(
            structure
        ):

            raise ValueError(
                f"Graph/structure node mismatch for "
                f"{configuration_id}: "
                f"graph={graph.num_nodes}, "
                f"structure={len(structure)}."
            )

        validate_graph_tensor_finiteness(
            graph
        )

    return True


def validate_graph_dataset_ids(
    graphs,
) -> bool:
    """
    Ensure graph configuration IDs are present and unique.
    """

    ids = [
        getattr(
            graph,
            "configuration_id",
            None,
        )
        for graph in graphs
    ]

    if any(
        configuration_id is None
        for configuration_id in ids
    ):

        raise ValueError(
            "At least one graph is missing configuration_id."
        )

    if len(ids) != len(
        set(ids)
    ):

        raise ValueError(
            "Duplicate graph configuration IDs detected."
        )

    return True


# =====================================================================
# Cross-object consistency
# =====================================================================

def validate_pipeline_alignment(
    structure_dataframe: pd.DataFrame,
    feature_dataframe: pd.DataFrame,
    graphs,
    configuration_column: str = "configuration_id",
) -> dict:
    """
    Validate alignment across structures, features, and graphs.
    """

    structure_ids = set(
        structure_dataframe[
            configuration_column
        ]
    )

    feature_ids = set(
        feature_dataframe[
            configuration_column
        ]
    )

    graph_ids = {
        getattr(
            graph,
            "configuration_id",
            None,
        )
        for graph in graphs
    }

    structure_feature_match = (
        structure_ids
        == feature_ids
    )

    structure_graph_match = (
        structure_ids
        == graph_ids
    )

    feature_graph_match = (
        feature_ids
        == graph_ids
    )

    return {
        "structure_count":
            len(
                structure_ids
            ),

        "feature_count":
            len(
                feature_ids
            ),

        "graph_count":
            len(
                graph_ids
            ),

        "structure_feature_match":
            bool(
                structure_feature_match
            ),

        "structure_graph_match":
            bool(
                structure_graph_match
            ),

        "feature_graph_match":
            bool(
                feature_graph_match
            ),

        "fully_aligned":
            bool(
                structure_feature_match
                and structure_graph_match
                and feature_graph_match
            ),
    }
