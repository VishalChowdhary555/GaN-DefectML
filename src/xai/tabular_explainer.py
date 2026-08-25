"""
Tabular explainability utilities for GaN-DefectML.

This module provides:
- permutation feature importance,
- physics-group aggregation,
- SHAP explanations,
- descriptor-group ablation,
- and XAI readiness checks.

Explainability routines are only executed when a valid trained
supervised model and labeled evaluation data are available.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


DEFAULT_RANDOM_SEED = 42
DEFAULT_PERMUTATION_REPEATS = 30


# ---------------------------------------------------------------------
# XAI readiness
# ---------------------------------------------------------------------

def check_tabular_xai_readiness(
    model=None,
    X=None,
    y=None,
) -> dict:
    """
    Check whether tabular explainability can be performed.
    """

    model_available = (
        model is not None
    )

    features_available = (
        X is not None
        and len(X) > 0
    )

    targets_available = (
        y is not None
        and len(y) > 0
    )

    fitted_model = False

    if model_available:

        fitted_model = any(
            attribute.endswith("_")
            for attribute in vars(
                model
            )
        )

    is_ready = bool(
        model_available
        and features_available
        and targets_available
        and fitted_model
    )

    return {
        "is_ready":
            is_ready,

        "model_available":
            bool(
                model_available
            ),

        "model_appears_fitted":
            bool(
                fitted_model
            ),

        "features_available":
            bool(
                features_available
            ),

        "targets_available":
            bool(
                targets_available
            ),
    }


# ---------------------------------------------------------------------
# Permutation importance
# ---------------------------------------------------------------------

def calculate_permutation_importance(
    model,
    X: pd.DataFrame,
    y,
    scoring: str = "neg_mean_absolute_error",
    n_repeats: int = DEFAULT_PERMUTATION_REPEATS,
    random_state: int = DEFAULT_RANDOM_SEED,
    n_jobs: Optional[int] = None,
) -> pd.DataFrame:
    """
    Calculate feature-level permutation importance.

    Parameters
    ----------
    model
        Fitted scikit-learn-compatible estimator.

    X
        Numerical feature matrix.

    y
        Validated target vector.

    scoring
        Scikit-learn scoring metric.

    n_repeats
        Number of independent feature permutations.

    Returns
    -------
    pandas.DataFrame
        Ranked feature importance table.
    """

    readiness = (
        check_tabular_xai_readiness(
            model=model,
            X=X,
            y=y,
        )
    )

    if not readiness[
        "is_ready"
    ]:

        raise RuntimeError(
            "Permutation importance cannot be "
            "calculated because tabular XAI "
            "requirements are not satisfied."
        )

    result = permutation_importance(
        estimator=model,
        X=X,
        y=y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    importance_df = pd.DataFrame(
        {
            "feature":
                X.columns,

            "importance_mean":
                result.importances_mean,

            "importance_std":
                result.importances_std,
        }
    )

    importance_df[
        "absolute_importance"
    ] = np.abs(
        importance_df[
            "importance_mean"
        ]
    )

    return (
        importance_df
        .sort_values(
            "absolute_importance",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


# ---------------------------------------------------------------------
# Physics-informed feature grouping
# ---------------------------------------------------------------------

DEFAULT_FEATURE_GROUPS = {
    "composition": [
        "atomic_fraction",
        "dopant_fraction",
        "Ga_fraction",
        "N_fraction",
        "Mg_fraction",
        "Si_fraction",
        "O_fraction",
        "C_fraction",
    ],

    "structure": [
        "volume",
        "density",
        "lattice",
        "coordination",
        "neighbor",
        "distance",
        "bond",
    ],

    "defect_identity": [
        "defect",
        "vacancy",
        "interstitial",
        "antisite",
        "substitution",
        "operation",
    ],

    "chemical_difference": [
        "delta_",
        "normalized_delta_",
        "electronegativity",
        "ionization",
        "electron_affinity",
        "atomic_radius",
        "atomic_mass",
        "atomic_number",
        "valence",
        "group",
        "row",
    ],

    "local_environment": [
        "nearest",
        "shell",
        "local",
        "coordination",
        "neighbor",
        "distance",
    ],
}


def assign_feature_group(
    feature_name: str,
    feature_groups: Optional[
        Dict[str, Iterable[str]]
    ] = None,
) -> str:
    """
    Assign a feature to a physics-informed descriptor group.

    Matching is performed using case-insensitive substrings.
    """

    if feature_groups is None:
        feature_groups = (
            DEFAULT_FEATURE_GROUPS
        )

    normalized_name = (
        feature_name.lower()
    )

    for group_name, patterns in (
        feature_groups.items()
    ):

        for pattern in patterns:

            if (
                pattern.lower()
                in normalized_name
            ):

                return group_name

    return "other"


def add_feature_groups(
    importance_df: pd.DataFrame,
    feature_groups: Optional[
        Dict[str, Iterable[str]]
    ] = None,
) -> pd.DataFrame:
    """
    Add physics-group labels to a feature importance table.
    """

    if "feature" not in (
        importance_df.columns
    ):

        raise ValueError(
            "Importance table must contain "
            "a 'feature' column."
        )

    result = (
        importance_df.copy()
    )

    result[
        "feature_group"
    ] = [
        assign_feature_group(
            feature_name=feature,
            feature_groups=
                feature_groups,
        )
        for feature in result[
            "feature"
        ]
    ]

    return result


def aggregate_grouped_importance(
    importance_df: pd.DataFrame,
    feature_groups: Optional[
        Dict[str, Iterable[str]]
    ] = None,
) -> pd.DataFrame:
    """
    Aggregate feature-level importance into physics-informed groups.
    """

    grouped_df = (
        add_feature_groups(
            importance_df=
                importance_df,
            feature_groups=
                feature_groups,
        )
    )

    importance_column = (
        "absolute_importance"
        if "absolute_importance"
        in grouped_df.columns
        else "importance_mean"
    )

    summary = (
        grouped_df
        .groupby(
            "feature_group",
            as_index=False,
        )
        .agg(
            total_importance=(
                importance_column,
                "sum",
            ),

            mean_importance=(
                importance_column,
                "mean",
            ),

            number_of_features=(
                "feature",
                "count",
            ),
        )
    )

    total = (
        summary[
            "total_importance"
        ].sum()
    )

    if total > 0:

        summary[
            "importance_fraction"
        ] = (
            summary[
                "total_importance"
            ]
            / total
        )

    else:

        summary[
            "importance_fraction"
        ] = 0.0

    return (
        summary
        .sort_values(
            "total_importance",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


# ---------------------------------------------------------------------
# Regression scoring helper
# ---------------------------------------------------------------------

def regression_score_summary(
    y_true,
    y_pred,
) -> dict:
    """
    Calculate regression metrics used by descriptor ablation.
    """

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    return {
        "MAE":
            float(
                mean_absolute_error(
                    y_true,
                    y_pred,
                )
            ),

        "RMSE":
            float(
                np.sqrt(
                    mean_squared_error(
                        y_true,
                        y_pred,
                    )
                )
            ),

        "R2":
            float(
                r2_score(
                    y_true,
                    y_pred,
                )
            )
            if len(
                y_true
            ) >= 2
            else np.nan,
    }


# ---------------------------------------------------------------------
# Descriptor-group ablation
# ---------------------------------------------------------------------

def calculate_descriptor_ablation(
    model,
    X: pd.DataFrame,
    y,
    feature_groups: Optional[
        Dict[str, Iterable[str]]
    ] = None,
    baseline: str = "mean",
) -> pd.DataFrame:
    """
    Measure model sensitivity to complete descriptor groups.

    Each physics-informed feature group is replaced by a baseline and
    the resulting degradation in prediction quality is measured.

    This differs from permutation importance because entire physical
    descriptor families are perturbed simultaneously.
    """

    readiness = (
        check_tabular_xai_readiness(
            model=model,
            X=X,
            y=y,
        )
    )

    if not readiness[
        "is_ready"
    ]:

        raise RuntimeError(
            "Descriptor ablation cannot be "
            "performed because no valid trained "
            "tabular model is available."
        )

    if feature_groups is None:
        feature_groups = (
            DEFAULT_FEATURE_GROUPS
        )

    original_prediction = (
        model.predict(
            X
        )
    )

    baseline_metrics = (
        regression_score_summary(
            y_true=y,
            y_pred=
                original_prediction,
        )
    )

    feature_group_map = {
        feature:
            assign_feature_group(
                feature,
                feature_groups,
            )
        for feature in X.columns
    }

    groups = sorted(
        set(
            feature_group_map.values()
        )
    )

    records = []

    for group in groups:

        group_features = [
            feature
            for feature, assigned_group
            in feature_group_map.items()
            if assigned_group == group
        ]

        if not group_features:
            continue

        ablated_X = (
            X.copy()
        )

        if baseline == "mean":

            for feature in (
                group_features
            ):

                ablated_X[
                    feature
                ] = X[
                    feature
                ].mean()

        elif baseline == "zero":

            ablated_X[
                group_features
            ] = 0.0

        else:

            raise ValueError(
                "baseline must be either "
                "'mean' or 'zero'."
            )

        ablated_prediction = (
            model.predict(
                ablated_X
            )
        )

        ablated_metrics = (
            regression_score_summary(
                y_true=y,
                y_pred=
                    ablated_prediction,
            )
        )

        records.append(
            {
                "feature_group":
                    group,

                "number_of_features":
                    len(
                        group_features
                    ),

                "baseline_MAE":
                    baseline_metrics[
                        "MAE"
                    ],

                "ablated_MAE":
                    ablated_metrics[
                        "MAE"
                    ],

                "MAE_increase":
                    (
                        ablated_metrics[
                            "MAE"
                        ]
                        -
                        baseline_metrics[
                            "MAE"
                        ]
                    ),

                "baseline_RMSE":
                    baseline_metrics[
                        "RMSE"
                    ],

                "ablated_RMSE":
                    ablated_metrics[
                        "RMSE"
                    ],

                "RMSE_increase":
                    (
                        ablated_metrics[
                            "RMSE"
                        ]
                        -
                        baseline_metrics[
                            "RMSE"
                        ]
                    ),

                "baseline_R2":
                    baseline_metrics[
                        "R2"
                    ],

                "ablated_R2":
                    ablated_metrics[
                        "R2"
                    ],
            }
        )

    if not records:

        return pd.DataFrame()

    return (
        pd.DataFrame(
            records
        )
        .sort_values(
            "MAE_increase",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


# ---------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------

def shap_is_available() -> bool:
    """
    Check whether SHAP is installed.
    """

    try:
        import shap  # noqa: F401

        return True

    except ImportError:
        return False


def calculate_shap_values(
    model,
    X: pd.DataFrame,
    background_X: Optional[
        pd.DataFrame
    ] = None,
):
    """
    Generate SHAP values for a fitted tabular model.

    SHAP is imported lazily so the rest of the repository can be used
    without requiring SHAP at import time.

    Returns
    -------
    tuple
        ``(explainer, shap_values)``
    """

    readiness = (
        check_tabular_xai_readiness(
            model=model,
            X=X,
            y=np.ones(
                len(X)
            ),
        )
    )

    if not readiness[
        "is_ready"
    ]:

        raise RuntimeError(
            "SHAP explanation cannot be "
            "performed because the model "
            "is not ready."
        )

    if not shap_is_available():

        raise ImportError(
            "SHAP is not installed. "
            "Install the optional XAI "
            "dependencies first."
        )

    import shap

    if background_X is None:

        background_X = X

    try:

        explainer = (
            shap.Explainer(
                model,
                background_X,
            )
        )

        shap_values = (
            explainer(
                X
            )
        )

    except Exception:

        # Generic model-agnostic fallback.
        prediction_function = (
            model.predict
        )

        explainer = (
            shap.Explainer(
                prediction_function,
                background_X,
            )
        )

        shap_values = (
            explainer(
                X
            )
        )

    return (
        explainer,
        shap_values,
    )


def build_shap_importance_table(
    shap_values,
    feature_names: Iterable[str],
) -> pd.DataFrame:
    """
    Convert SHAP values into mean absolute feature importance.
    """

    feature_names = list(
        feature_names
    )

    values = np.asarray(
        shap_values.values
    )

    # Handle single-output explanations.
    if values.ndim == 3:

        values = values[
            :,
            :,
            0
        ]

    if values.ndim != 2:

        raise ValueError(
            "Expected a two-dimensional "
            "SHAP value matrix."
        )

    if values.shape[1] != (
        len(
            feature_names
        )
    ):

        raise ValueError(
            "Number of SHAP feature columns "
            "does not match feature names."
        )

    mean_absolute_shap = (
        np.mean(
            np.abs(
                values
            ),
            axis=0,
        )
    )

    importance_df = pd.DataFrame(
        {
            "feature":
                feature_names,

            "mean_absolute_SHAP":
                mean_absolute_shap,
        }
    )

    return (
        importance_df
        .sort_values(
            "mean_absolute_SHAP",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


# ---------------------------------------------------------------------
# High-level XAI pipeline
# ---------------------------------------------------------------------

def run_tabular_xai(
    model,
    X: pd.DataFrame,
    y,
    feature_groups: Optional[
        Dict[str, Iterable[str]]
    ] = None,
    permutation_repeats: int = DEFAULT_PERMUTATION_REPEATS,
    random_state: int = DEFAULT_RANDOM_SEED,
    run_shap: bool = True,
) -> Optional[dict]:
    """
    Run the complete tabular explainability pipeline.

    Returns None when no validated trained model exists.
    """

    readiness = (
        check_tabular_xai_readiness(
            model=model,
            X=X,
            y=y,
        )
    )

    if not readiness[
        "is_ready"
    ]:

        print(
            "Tabular model explanation is inactive "
            "because no validated trained model exists."
        )

        return None

    permutation_df = (
        calculate_permutation_importance(
            model=model,
            X=X,
            y=y,
            n_repeats=
                permutation_repeats,
            random_state=
                random_state,
        )
    )

    grouped_df = (
        aggregate_grouped_importance(
            importance_df=
                permutation_df,
            feature_groups=
                feature_groups,
        )
    )

    ablation_df = (
        calculate_descriptor_ablation(
            model=model,
            X=X,
            y=y,
            feature_groups=
                feature_groups,
        )
    )

    shap_result = None
    shap_importance_df = None

    if (
        run_shap
        and shap_is_available()
    ):

        explainer, shap_values = (
            calculate_shap_values(
                model=model,
                X=X,
            )
        )

        shap_importance_df = (
            build_shap_importance_table(
                shap_values=
                    shap_values,
                feature_names=
                    X.columns,
            )
        )

        shap_result = {
            "explainer":
                explainer,

            "values":
                shap_values,
        }

    return {
        "readiness":
            readiness,

        "permutation_importance":
            permutation_df,

        "grouped_importance":
            grouped_df,

        "descriptor_ablation":
            ablation_df,

        "shap":
            shap_result,

        "shap_importance":
            shap_importance_df,
    }
