"""
Evaluation and reporting utilities for GaN-DefectML.

This module provides:
- regression and classification metric summaries,
- prediction tables,
- fold-level aggregation,
- model-ranking helpers,
- and export utilities for evaluation results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ---------------------------------------------------------------------
# Regression evaluation
# ---------------------------------------------------------------------

def regression_metrics(
    y_true,
    y_pred,
) -> dict:
    """
    Calculate standard regression metrics.

    Returns
    -------
    dict
        MAE, RMSE, and R2.
    """

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape."
        )

    if y_true.size == 0:
        raise ValueError(
            "Regression metric calculation requires "
            "at least one sample."
        )

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = (
        r2_score(
            y_true,
            y_pred,
        )
        if len(y_true) >= 2
        else np.nan
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


def build_regression_prediction_table(
    configuration_ids,
    y_true,
    y_pred,
) -> pd.DataFrame:
    """
    Build a configuration-level regression prediction table.
    """

    configuration_ids = list(
        configuration_ids
    )

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    if not (
        len(configuration_ids)
        == len(y_true)
        == len(y_pred)
    ):
        raise ValueError(
            "configuration_ids, y_true, and y_pred "
            "must have equal lengths."
        )

    prediction_df = pd.DataFrame(
        {
            "configuration_id":
                configuration_ids,

            "true_value":
                y_true,

            "predicted_value":
                y_pred,
        }
    )

    prediction_df[
        "signed_error"
    ] = (
        prediction_df[
            "predicted_value"
        ]
        -
        prediction_df[
            "true_value"
        ]
    )

    prediction_df[
        "absolute_error"
    ] = np.abs(
        prediction_df[
            "signed_error"
        ]
    )

    return prediction_df


# ---------------------------------------------------------------------
# Classification evaluation
# ---------------------------------------------------------------------

def classification_metrics(
    y_true,
    y_pred,
) -> dict:
    """
    Calculate standard classification metrics.

    Returns
    -------
    dict
        Accuracy, balanced accuracy, macro F1, and weighted F1.
    """

    y_true = np.asarray(
        y_true,
    )

    y_pred = np.asarray(
        y_pred,
    )

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape."
        )

    if y_true.size == 0:
        raise ValueError(
            "Classification metric calculation requires "
            "at least one sample."
        )

    return {
        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    y_pred,
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y_true,
                    y_pred,
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                )
            ),

        "weighted_f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )
            ),
    }


def build_classification_prediction_table(
    configuration_ids,
    y_true,
    y_pred,
) -> pd.DataFrame:
    """
    Build a configuration-level classification prediction table.
    """

    configuration_ids = list(
        configuration_ids
    )

    y_true = np.asarray(
        y_true,
    )

    y_pred = np.asarray(
        y_pred,
    )

    if not (
        len(configuration_ids)
        == len(y_true)
        == len(y_pred)
    ):
        raise ValueError(
            "configuration_ids, y_true, and y_pred "
            "must have equal lengths."
        )

    prediction_df = pd.DataFrame(
        {
            "configuration_id":
                configuration_ids,

            "true_label":
                y_true,

            "predicted_label":
                y_pred,
        }
    )

    prediction_df[
        "is_correct"
    ] = (
        prediction_df[
            "true_label"
        ]
        ==
        prediction_df[
            "predicted_label"
        ]
    )

    return prediction_df


# ---------------------------------------------------------------------
# Fold-level summaries
# ---------------------------------------------------------------------

def summarize_fold_metrics(
    fold_metrics_df: pd.DataFrame,
    fold_column: str = "fold",
) -> pd.DataFrame:
    """
    Summarize mean and standard deviation of fold-level metrics.

    Parameters
    ----------
    fold_metrics_df
        Table containing one row per CV fold.

    fold_column
        Fold identifier column.

    Returns
    -------
    pandas.DataFrame
        Metric, mean, standard deviation, minimum, and maximum.
    """

    if fold_metrics_df.empty:
        return pd.DataFrame(
            columns=[
                "metric",
                "mean",
                "std",
                "min",
                "max",
            ]
        )

    numeric_columns = [
        column
        for column in fold_metrics_df.columns
        if (
            column != fold_column
            and pd.api.types.is_numeric_dtype(
                fold_metrics_df[
                    column
                ]
            )
        )
    ]

    records = []

    for metric in numeric_columns:

        values = pd.to_numeric(
            fold_metrics_df[
                metric
            ],
            errors="coerce",
        ).dropna()

        if values.empty:
            continue

        records.append(
            {
                "metric":
                    metric,

                "mean":
                    float(
                        values.mean()
                    ),

                "std":
                    float(
                        values.std(
                            ddof=0
                        )
                    ),

                "min":
                    float(
                        values.min()
                    ),

                "max":
                    float(
                        values.max()
                    ),
            }
        )

    return pd.DataFrame(
        records
    )


# ---------------------------------------------------------------------
# Model ranking
# ---------------------------------------------------------------------

def rank_regression_models(
    model_comparison_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rank regression models by MAE, then RMSE, then R2.
    """

    required_columns = {
        "model",
        "MAE",
        "RMSE",
        "R2",
    }

    missing = (
        required_columns
        - set(
            model_comparison_df.columns
        )
    )

    if missing:
        raise ValueError(
            "Regression comparison table is missing: "
            f"{sorted(missing)}"
        )

    ranked = (
        model_comparison_df
        .sort_values(
            by=[
                "MAE",
                "RMSE",
                "R2",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    ranked.insert(
        0,
        "rank",
        np.arange(
            1,
            len(ranked) + 1,
        ),
    )

    return ranked


def rank_classification_models(
    model_comparison_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rank classification models by macro F1 and balanced accuracy.
    """

    required_columns = {
        "model",
        "macro_f1",
        "balanced_accuracy",
        "accuracy",
    }

    missing = (
        required_columns
        - set(
            model_comparison_df.columns
        )
    )

    if missing:
        raise ValueError(
            "Classification comparison table is missing: "
            f"{sorted(missing)}"
        )

    ranked = (
        model_comparison_df
        .sort_values(
            by=[
                "macro_f1",
                "balanced_accuracy",
                "accuracy",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    ranked.insert(
        0,
        "rank",
        np.arange(
            1,
            len(ranked) + 1,
        ),
    )

    return ranked


# ---------------------------------------------------------------------
# General model-report helper
# ---------------------------------------------------------------------

def build_evaluation_report(
    problem_type: str,
    configuration_ids,
    y_true,
    y_pred,
    model_name: Optional[str] = None,
) -> dict:
    """
    Build a complete evaluation report for one model.

    Parameters
    ----------
    problem_type
        ``"regression"`` or ``"classification"``.

    configuration_ids
        Sample identifiers.

    y_true, y_pred
        Ground-truth and predicted values.

    model_name
        Optional model label.

    Returns
    -------
    dict
        Metrics and configuration-level prediction table.
    """

    problem_type = (
        problem_type
        .strip()
        .lower()
    )

    if problem_type == "regression":

        metrics = regression_metrics(
            y_true=y_true,
            y_pred=y_pred,
        )

        prediction_df = (
            build_regression_prediction_table(
                configuration_ids=
                    configuration_ids,
                y_true=y_true,
                y_pred=y_pred,
            )
        )

    elif problem_type == "classification":

        metrics = classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
        )

        prediction_df = (
            build_classification_prediction_table(
                configuration_ids=
                    configuration_ids,
                y_true=y_true,
                y_pred=y_pred,
            )
        )

    else:
        raise ValueError(
            "problem_type must be either "
            "'regression' or 'classification'."
        )

    return {
        "model_name":
            model_name,

        "problem_type":
            problem_type,

        "metrics":
            metrics,

        "predictions":
            prediction_df,
    }


# ---------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------

def export_evaluation_table(
    dataframe: pd.DataFrame,
    output_path: str | Path,
    index: bool = False,
) -> Path:
    """
    Save an evaluation table as CSV.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output_path,
        index=index,
    )

    return output_path


def export_evaluation_report(
    report: dict,
    output_directory: str | Path,
    prefix: str,
) -> dict:
    """
    Save metrics and predictions from an evaluation report.

    Returns
    -------
    dict
        Paths of the generated files.
    """

    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path = (
        output_directory
        / f"{prefix}_metrics.csv"
    )

    predictions_path = (
        output_directory
        / f"{prefix}_predictions.csv"
    )

    metrics_df = pd.DataFrame(
        [
            {
                "model_name":
                    report.get(
                        "model_name"
                    ),

                "problem_type":
                    report[
                        "problem_type"
                    ],

                **report[
                    "metrics"
                ],
            }
        ]
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    report[
        "predictions"
    ].to_csv(
        predictions_path,
        index=False,
    )

    return {
        "metrics_path":
            metrics_path,

        "predictions_path":
            predictions_path,
    }
