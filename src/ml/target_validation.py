"""
Target validation utilities for GaN-DefectML.

This module defines:
- target schema conventions,
- target-table validation,
- label completeness audits,
- regression/classification readiness checks,
- and configuration-level target aggregation.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


TARGET_SCHEMA_COLUMNS = [
    "configuration_id",
    "host_material_id",
    "host_phase",
    "calculation_id",
    "data_source",
    "source_reference",
    "calculation_method",
    "exchange_correlation_functional",
    "charge_state",
    "chemical_potential_condition",
    "fermi_level_eV",
    "temperature_K",
    "formation_energy_eV",
    "relaxed_band_gap_eV",
    "band_gap_change_eV",
    "transition_level_eV",
    "carrier_type",
    "donor_acceptor_class",
    "carrier_concentration_cm3",
    "is_relaxed",
    "relaxation_energy_eV",
    "maximum_displacement_A",
    "volume_change_percent",
    "label_quality",
    "convergence_status",
    "notes",
]


REGRESSION_TARGETS = [
    "formation_energy_eV",
    "relaxed_band_gap_eV",
    "band_gap_change_eV",
    "transition_level_eV",
    "carrier_concentration_cm3",
]


CLASSIFICATION_TARGETS = [
    "donor_acceptor_class",
    "carrier_type",
]


VALID_CARRIER_TYPES = {
    "n-type",
    "p-type",
    "compensating",
    "neutral",
    "unknown",
}


VALID_DONOR_ACCEPTOR_CLASSES = {
    "donor",
    "acceptor",
    "amphoteric",
    "compensating",
    "inactive",
    "unknown",
}


VALID_LABEL_QUALITIES = {
    "high",
    "medium",
    "low",
    "unverified",
}


def create_blank_target_template(
    configuration_ids: Iterable[str],
    host_material_id: Optional[str] = None,
    host_phase: str = "Wurtzite",
) -> pd.DataFrame:
    """
    Create a blank target-label template.

    Parameters
    ----------
    configuration_ids
        Structure configuration identifiers.

    host_material_id
        Optional host material identifier.

    host_phase
        Host phase label.

    Returns
    -------
    pandas.DataFrame
        Blank target table.
    """

    records = []

    for configuration_id in configuration_ids:

        record = {
            column: None
            for column
            in TARGET_SCHEMA_COLUMNS
        }

        record["configuration_id"] = (
            configuration_id
        )

        record["host_material_id"] = (
            host_material_id
        )

        record["host_phase"] = (
            host_phase
        )

        # Unknown until a calculation is supplied.
        record["is_relaxed"] = None

        records.append(record)

    return pd.DataFrame(
        records,
        columns=TARGET_SCHEMA_COLUMNS,
    )


def validate_target_table(
    target_df: pd.DataFrame,
    valid_configuration_ids: Iterable[str],
) -> list[str]:
    """
    Validate target identifiers, metadata, and numerical fields.

    Returns
    -------
    list[str]
        Non-fatal validation warnings.

    Raises
    ------
    ValueError
        If required columns or configuration IDs are invalid.
    """

    validation_messages = []

    required_columns = {
        "configuration_id",
        "data_source",
        "charge_state",
        "formation_energy_eV",
        "is_relaxed",
    }

    missing_columns = (
        required_columns
        - set(target_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Target table is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    valid_configuration_ids = set(
        valid_configuration_ids
    )

    invalid_configuration_ids = set(
        target_df[
            "configuration_id"
        ]
        .dropna()
    ) - valid_configuration_ids

    if invalid_configuration_ids:
        raise ValueError(
            "Unknown configuration IDs found: "
            f"{sorted(invalid_configuration_ids)}"
        )

    if "calculation_id" in target_df.columns:

        calculation_ids = (
            target_df[
                "calculation_id"
            ]
            .dropna()
        )

        duplicated = (
            calculation_ids[
                calculation_ids.duplicated()
            ]
            .unique()
            .tolist()
        )

        if duplicated:
            validation_messages.append(
                "Duplicate calculation IDs detected: "
                f"{duplicated}"
            )

    if "carrier_type" in target_df.columns:

        invalid_values = set(
            target_df[
                "carrier_type"
            ]
            .dropna()
        ) - VALID_CARRIER_TYPES

        if invalid_values:
            validation_messages.append(
                "Unexpected carrier types: "
                f"{sorted(invalid_values)}"
            )

    if (
        "donor_acceptor_class"
        in target_df.columns
    ):

        invalid_values = set(
            target_df[
                "donor_acceptor_class"
            ]
            .dropna()
        ) - VALID_DONOR_ACCEPTOR_CLASSES

        if invalid_values:
            validation_messages.append(
                "Unexpected donor/acceptor classes: "
                f"{sorted(invalid_values)}"
            )

    if "label_quality" in target_df.columns:

        invalid_values = set(
            target_df[
                "label_quality"
            ]
            .dropna()
        ) - VALID_LABEL_QUALITIES

        if invalid_values:
            validation_messages.append(
                "Unexpected label-quality values: "
                f"{sorted(invalid_values)}"
            )

    numeric_columns = [
        "charge_state",
        "fermi_level_eV",
        "temperature_K",
        "formation_energy_eV",
        "relaxed_band_gap_eV",
        "band_gap_change_eV",
        "transition_level_eV",
        "carrier_concentration_cm3",
        "relaxation_energy_eV",
        "maximum_displacement_A",
        "volume_change_percent",
    ]

    for column in numeric_columns:

        if column not in target_df.columns:
            continue

        converted = pd.to_numeric(
            target_df[column],
            errors="coerce",
        )

        invalid_count = (
            target_df[column].notna()
            & converted.isna()
        ).sum()

        if invalid_count > 0:

            validation_messages.append(
                f"{column}: "
                f"{int(invalid_count)} non-numeric values"
            )

    return validation_messages


def audit_target_completeness(
    target_df: pd.DataFrame,
    target_columns: Optional[
        Iterable[str]
    ] = None,
) -> pd.DataFrame:
    """
    Summarize target-label completeness.
    """

    if target_columns is None:
        target_columns = [
            *REGRESSION_TARGETS,
            *CLASSIFICATION_TARGETS,
        ]

    records = []

    for target in target_columns:

        if target not in target_df.columns:
            continue

        available = int(
            target_df[
                target
            ].notna().sum()
        )

        unique_values = int(
            target_df[
                target
            ]
            .dropna()
            .nunique()
        )

        records.append(
            {
                "target":
                    target,

                "available_labels":
                    available,

                "missing_labels":
                    int(
                        len(target_df)
                        - available
                    ),

                "unique_values":
                    unique_values,

                "availability_fraction":
                    (
                        available
                        / len(target_df)
                        if len(target_df) > 0
                        else np.nan
                    ),
            }
        )

    return pd.DataFrame(
        records
    )


def assess_regression_readiness(
    target_series: pd.Series,
    minimum_samples: int = 30,
    minimum_unique_values: int = 5,
) -> dict:
    """
    Assess whether a regression target is ready for baseline modeling.
    """

    clean_target = (
        pd.to_numeric(
            target_series,
            errors="coerce",
        )
        .dropna()
    )

    return {
        "available_samples":
            int(
                len(clean_target)
            ),

        "unique_values":
            int(
                clean_target.nunique()
            ),

        "minimum_required":
            int(
                minimum_samples
            ),

        "is_ready":
            bool(
                len(clean_target)
                >= minimum_samples
                and clean_target.nunique()
                >= minimum_unique_values
            ),
    }


def assess_classification_readiness(
    target_series: pd.Series,
    minimum_samples: int = 30,
    minimum_samples_per_class: int = 5,
) -> dict:
    """
    Assess whether a classification target is ready for baseline modeling.
    """

    clean_target = (
        target_series
        .dropna()
    )

    class_counts = (
        clean_target
        .value_counts()
    )

    minimum_class_count = (
        int(
            class_counts.min()
        )
        if not class_counts.empty
        else 0
    )

    return {
        "available_samples":
            int(
                len(clean_target)
            ),

        "number_of_classes":
            int(
                clean_target.nunique()
            ),

        "minimum_class_count":
            minimum_class_count,

        "is_ready":
            bool(
                len(clean_target)
                >= minimum_samples
                and clean_target.nunique()
                >= 2
                and minimum_class_count
                >= minimum_samples_per_class
            ),
    }


def build_target_readiness_table(
    target_df: pd.DataFrame,
    minimum_regression_samples: int = 30,
    minimum_classification_samples: int = 30,
    minimum_samples_per_class: int = 5,
) -> pd.DataFrame:
    """
    Evaluate readiness for all supported supervised targets.
    """

    records = []

    for target in REGRESSION_TARGETS:

        if target not in target_df.columns:
            continue

        result = assess_regression_readiness(
            target_df[target],
            minimum_samples=
                minimum_regression_samples,
        )

        records.append(
            {
                "target":
                    target,

                "problem_type":
                    "Regression",

                "available_samples":
                    result[
                        "available_samples"
                    ],

                "unique_values":
                    result[
                        "unique_values"
                    ],

                "minimum_required":
                    result[
                        "minimum_required"
                    ],

                "minimum_class_count":
                    np.nan,

                "is_ready":
                    result[
                        "is_ready"
                    ],
            }
        )

    for target in CLASSIFICATION_TARGETS:

        if target not in target_df.columns:
            continue

        result = (
            assess_classification_readiness(
                target_df[target],
                minimum_samples=
                    minimum_classification_samples,
                minimum_samples_per_class=
                    minimum_samples_per_class,
            )
        )

        records.append(
            {
                "target":
                    target,

                "problem_type":
                    "Classification",

                "available_samples":
                    result[
                        "available_samples"
                    ],

                "unique_values":
                    result[
                        "number_of_classes"
                    ],

                "minimum_required":
                    minimum_classification_samples,

                "minimum_class_count":
                    result[
                        "minimum_class_count"
                    ],

                "is_ready":
                    result[
                        "is_ready"
                    ],
            }
        )

    return pd.DataFrame(
        records
    )


def build_configuration_target_table(
    target_df: pd.DataFrame,
    target_column: str,
    aggregation_method: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convert calculation-level labels into one target per structure.

    Multiple labels for one structure are rejected unless an explicit
    numerical aggregation strategy is provided.

    Parameters
    ----------
    aggregation_method
        One of:
        ``None``, ``mean``, ``median``, ``minimum``, or ``maximum``.
    """

    if target_column not in target_df.columns:
        raise ValueError(
            f"Target column '{target_column}' "
            "does not exist."
        )

    working_df = target_df[
        [
            "configuration_id",
            target_column,
        ]
    ].copy()

    if target_column in REGRESSION_TARGETS:

        working_df[
            target_column
        ] = pd.to_numeric(
            working_df[
                target_column
            ],
            errors="coerce",
        )

    working_df = (
        working_df.dropna(
            subset=[
                target_column
            ]
        )
    )

    duplicate_counts = (
        working_df[
            "configuration_id"
        ]
        .value_counts()
    )

    duplicated_ids = (
        duplicate_counts[
            duplicate_counts > 1
        ]
        .index
        .tolist()
    )

    if (
        duplicated_ids
        and aggregation_method is None
    ):

        raise ValueError(
            "Multiple labels exist for configurations: "
            f"{duplicated_ids}. "
            "Use an explicit aggregation strategy or "
            "model the calculation conditions separately."
        )

    if aggregation_method is None:

        return (
            working_df
            .drop_duplicates(
                subset=[
                    "configuration_id"
                ]
            )
            .reset_index(
                drop=True
            )
        )

    if target_column not in (
        REGRESSION_TARGETS
    ):

        raise ValueError(
            "Numerical aggregation is supported only "
            "for regression targets."
        )

    aggregation_functions = {
        "mean":
            "mean",

        "median":
            "median",

        "minimum":
            "min",

        "maximum":
            "max",
    }

    if (
        aggregation_method
        not in aggregation_functions
    ):

        raise ValueError(
            "Unsupported aggregation method: "
            f"{aggregation_method}"
        )

    return (
        working_df
        .groupby(
            "configuration_id",
            as_index=False,
        )[target_column]
        .agg(
            aggregation_functions[
                aggregation_method
            ]
        )
    )
