"""
Supervised training utilities for GaN-DefectML.

This module provides:
- supervised dataset assembly,
- scientific readiness gating,
- cross-validated regression,
- cross-validated classification,
- metric calculation,
- model comparison,
- and final model fitting.

The pipeline intentionally remains inactive when validated target
coverage is insufficient.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
)


DEFAULT_RANDOM_SEED = 42


# ---------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------

def assemble_supervised_dataset(
    feature_df: pd.DataFrame,
    target_df: pd.DataFrame,
    target_column: str,
    configuration_id_column: str = "configuration_id",
) -> pd.DataFrame:
    """
    Merge compact features with one supervised target.

    Only rows containing a valid target are retained.
    """

    if configuration_id_column not in feature_df.columns:
        raise ValueError(
            f"Feature table must contain "
            f"'{configuration_id_column}'."
        )

    if configuration_id_column not in target_df.columns:
        raise ValueError(
            f"Target table must contain "
            f"'{configuration_id_column}'."
        )

    if target_column not in target_df.columns:
        raise ValueError(
            f"Target column '{target_column}' "
            "was not found."
        )

    target_block = target_df[
        [
            configuration_id_column,
            target_column,
        ]
    ].copy()

    # One structure should correspond to one target at this stage.
    if target_block[
        configuration_id_column
    ].duplicated().any():

        duplicated_ids = (
            target_block.loc[
                target_block[
                    configuration_id_column
                ].duplicated(
                    keep=False
                ),
                configuration_id_column,
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Multiple target rows exist for "
            f"configurations: {duplicated_ids}. "
            "Resolve or aggregate calculation-level "
            "labels before training."
        )

    merged = feature_df.merge(
        target_block,
        on=configuration_id_column,
        how="inner",
        validate="one_to_one",
    )

    merged = merged.dropna(
        subset=[
            target_column
        ]
    ).reset_index(
        drop=True
    )

    return merged


def split_features_and_target(
    supervised_df: pd.DataFrame,
    target_column: str,
    configuration_id_column: str = "configuration_id",
):
    """
    Separate configuration IDs, predictors, and target values.
    """

    if target_column not in supervised_df.columns:
        raise ValueError(
            f"Target '{target_column}' "
            "not found."
        )

    configuration_ids = (
        supervised_df[
            configuration_id_column
        ].copy()
        if configuration_id_column
        in supervised_df.columns
        else pd.Series(
            np.arange(
                len(supervised_df)
            ),
            name="sample_id",
        )
    )

    X = supervised_df.drop(
        columns=[
            configuration_id_column,
            target_column,
        ],
        errors="ignore",
    ).copy()

    y = supervised_df[
        target_column
    ].copy()

    if X.empty:
        raise ValueError(
            "No predictor features remain."
        )

    non_numeric_columns = (
        X.select_dtypes(
            exclude=[
                np.number,
                "bool",
            ]
        )
        .columns
        .tolist()
    )

    if non_numeric_columns:
        raise ValueError(
            "Non-numerical predictors found: "
            f"{non_numeric_columns}"
        )

    X = X.astype(float)

    if np.isinf(
        X.to_numpy()
    ).any():

        raise ValueError(
            "Infinite values detected "
            "in predictor matrix."
        )

    return (
        configuration_ids,
        X,
        y,
    )


# ---------------------------------------------------------------------
# Scientific readiness gates
# ---------------------------------------------------------------------

def check_regression_readiness(
    y,
    minimum_samples: int = 30,
    minimum_unique_values: int = 5,
) -> dict:
    """
    Determine whether a regression target has enough valid labels.
    """

    y_numeric = pd.to_numeric(
        pd.Series(y),
        errors="coerce",
    ).dropna()

    n_samples = len(
        y_numeric
    )

    n_unique = (
        y_numeric.nunique()
    )

    ready = (
        n_samples >= minimum_samples
        and
        n_unique >= minimum_unique_values
    )

    return {
        "is_ready":
            bool(ready),

        "available_samples":
            int(n_samples),

        "unique_values":
            int(n_unique),

        "minimum_samples":
            int(minimum_samples),

        "minimum_unique_values":
            int(
                minimum_unique_values
            ),
    }


def check_classification_readiness(
    y,
    minimum_samples: int = 30,
    minimum_samples_per_class: int = 5,
) -> dict:
    """
    Determine whether a classification target has sufficient coverage.
    """

    y_clean = (
        pd.Series(y)
        .dropna()
    )

    class_counts = (
        y_clean
        .value_counts()
    )

    n_samples = len(
        y_clean
    )

    n_classes = len(
        class_counts
    )

    smallest_class = (
        int(
            class_counts.min()
        )
        if n_classes > 0
        else 0
    )

    ready = (
        n_samples >= minimum_samples
        and
        n_classes >= 2
        and
        smallest_class
        >= minimum_samples_per_class
    )

    return {
        "is_ready":
            bool(ready),

        "available_samples":
            int(n_samples),

        "number_of_classes":
            int(n_classes),

        "minimum_class_count":
            smallest_class,

        "minimum_samples":
            int(minimum_samples),

        "minimum_samples_per_class":
            int(
                minimum_samples_per_class
            ),

        "class_counts":
            class_counts.to_dict(),
    }


# ---------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------

def calculate_regression_metrics(
    y_true,
    y_pred,
) -> dict:
    """
    Calculate standard regression metrics.
    """

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
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
        "MAE":
            float(mae),

        "RMSE":
            float(rmse),

        "R2":
            float(r2),
    }


# ---------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------

def calculate_classification_metrics(
    y_true,
    y_pred,
) -> dict:
    """
    Calculate classification metrics suitable for potentially
    imbalanced defect-property classes.
    """

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


# ---------------------------------------------------------------------
# Regression cross-validation
# ---------------------------------------------------------------------

def cross_validate_regressor(
    model: BaseEstimator,
    X: pd.DataFrame,
    y,
    n_splits: int = 5,
    random_state: int = DEFAULT_RANDOM_SEED,
) -> dict:
    """
    Evaluate one regression model using K-fold cross-validation.
    """

    y = pd.to_numeric(
        pd.Series(y),
        errors="coerce",
    )

    if y.isna().any():
        raise ValueError(
            "Regression target contains "
            "missing/non-numeric values."
        )

    if n_splits > len(X):
        raise ValueError(
            "Number of CV folds exceeds "
            "the number of samples."
        )

    cv = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    predictions = np.full(
        len(X),
        np.nan,
        dtype=float,
    )

    fold_records = []

    for fold_index, (
        train_index,
        test_index,
    ) in enumerate(
        cv.split(X),
        start=1,
    ):

        fold_model = clone(
            model
        )

        X_train = X.iloc[
            train_index
        ]

        X_test = X.iloc[
            test_index
        ]

        y_train = y.iloc[
            train_index
        ]

        y_test = y.iloc[
            test_index
        ]

        fold_model.fit(
            X_train,
            y_train,
        )

        fold_prediction = (
            fold_model.predict(
                X_test
            )
        )

        predictions[
            test_index
        ] = fold_prediction

        fold_metrics = (
            calculate_regression_metrics(
                y_true=y_test,
                y_pred=fold_prediction,
            )
        )

        fold_records.append(
            {
                "fold":
                    fold_index,

                **fold_metrics,
            }
        )

    overall_metrics = (
        calculate_regression_metrics(
            y_true=y,
            y_pred=predictions,
        )
    )

    return {
        "metrics":
            overall_metrics,

        "fold_metrics":
            pd.DataFrame(
                fold_records
            ),

        "predictions":
            predictions,
    }


# ---------------------------------------------------------------------
# Classification cross-validation
# ---------------------------------------------------------------------

def cross_validate_classifier(
    model: BaseEstimator,
    X: pd.DataFrame,
    y,
    n_splits: int = 5,
    random_state: int = DEFAULT_RANDOM_SEED,
) -> dict:
    """
    Evaluate one classifier using stratified K-fold cross-validation.
    """

    y = (
        pd.Series(y)
        .reset_index(drop=True)
    )

    if y.isna().any():
        raise ValueError(
            "Classification target "
            "contains missing labels."
        )

    class_counts = (
        y.value_counts()
    )

    if (
        class_counts.min()
        < n_splits
    ):
        raise ValueError(
            "Every class must contain at "
            "least n_splits samples for "
            "stratified cross-validation."
        )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    predictions = np.empty(
        len(X),
        dtype=object,
    )

    fold_records = []

    for fold_index, (
        train_index,
        test_index,
    ) in enumerate(
        cv.split(
            X,
            y,
        ),
        start=1,
    ):

        fold_model = clone(
            model
        )

        X_train = X.iloc[
            train_index
        ]

        X_test = X.iloc[
            test_index
        ]

        y_train = y.iloc[
            train_index
        ]

        y_test = y.iloc[
            test_index
        ]

        fold_model.fit(
            X_train,
            y_train,
        )

        fold_prediction = (
            fold_model.predict(
                X_test
            )
        )

        predictions[
            test_index
        ] = fold_prediction

        fold_metrics = (
            calculate_classification_metrics(
                y_true=y_test,
                y_pred=fold_prediction,
            )
        )

        fold_records.append(
            {
                "fold":
                    fold_index,

                **fold_metrics,
            }
        )

    overall_metrics = (
        calculate_classification_metrics(
            y_true=y,
            y_pred=predictions,
        )
    )

    return {
        "metrics":
            overall_metrics,

        "fold_metrics":
            pd.DataFrame(
                fold_records
            ),

        "predictions":
            predictions,
    }


# ---------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------

def compare_regression_models(
    models: Dict[str, BaseEstimator],
    X: pd.DataFrame,
    y,
    n_splits: int = 5,
    random_state: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """
    Cross-validate and compare multiple regression models.
    """

    records = []

    for model_name, model in (
        models.items()
    ):

        result = (
            cross_validate_regressor(
                model=model,
                X=X,
                y=y,
                n_splits=n_splits,
                random_state=
                    random_state,
            )
        )

        records.append(
            {
                "model":
                    model_name,

                **result[
                    "metrics"
                ],
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            [
                "MAE",
                "RMSE",
            ],
            ascending=True,
        )
        .reset_index(drop=True)
    )


def compare_classification_models(
    models: Dict[str, BaseEstimator],
    X: pd.DataFrame,
    y,
    n_splits: int = 5,
    random_state: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """
    Cross-validate and compare multiple classification models.
    """

    records = []

    for model_name, model in (
        models.items()
    ):

        result = (
            cross_validate_classifier(
                model=model,
                X=X,
                y=y,
                n_splits=n_splits,
                random_state=
                    random_state,
            )
        )

        records.append(
            {
                "model":
                    model_name,

                **result[
                    "metrics"
                ],
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            [
                "macro_f1",
                "balanced_accuracy",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------
# Final model fitting
# ---------------------------------------------------------------------

def fit_final_model(
    model: BaseEstimator,
    X: pd.DataFrame,
    y,
) -> BaseEstimator:
    """
    Fit a fresh clone of the selected model on all validated samples.
    """

    final_model = clone(
        model
    )

    final_model.fit(
        X,
        y,
    )

    return final_model


# ---------------------------------------------------------------------
# High-level gated training
# ---------------------------------------------------------------------

def run_regression_training(
    models: Dict[str, BaseEstimator],
    feature_df: pd.DataFrame,
    target_df: pd.DataFrame,
    target_column: str,
    minimum_samples: int = 30,
    n_splits: int = 5,
    random_state: int = DEFAULT_RANDOM_SEED,
) -> Optional[dict]:
    """
    Run the complete gated regression workflow.

    Returns None when the target is not scientifically ready.
    """

    supervised_df = (
        assemble_supervised_dataset(
            feature_df=feature_df,
            target_df=target_df,
            target_column=
                target_column,
        )
    )

    (
        configuration_ids,
        X,
        y,
    ) = split_features_and_target(
        supervised_df=
            supervised_df,
        target_column=
            target_column,
    )

    readiness = (
        check_regression_readiness(
            y=y,
            minimum_samples=
                minimum_samples,
        )
    )

    if not readiness[
        "is_ready"
    ]:

        print(
            "Regression training skipped."
        )

        print(
            f"Available labels: "
            f"{readiness['available_samples']}"
        )

        print(
            f"Minimum required: "
            f"{readiness['minimum_samples']}"
        )

        return None

    comparison_df = (
        compare_regression_models(
            models=models,
            X=X,
            y=y,
            n_splits=n_splits,
            random_state=random_state,
        )
    )

    return {
        "target":
            target_column,

        "configuration_ids":
            configuration_ids,

        "X":
            X,

        "y":
            y,

        "readiness":
            readiness,

        "model_comparison":
            comparison_df,
    }


def run_classification_training(
    models: Dict[str, BaseEstimator],
    feature_df: pd.DataFrame,
    target_df: pd.DataFrame,
    target_column: str,
    minimum_samples: int = 30,
    minimum_samples_per_class: int = 5,
    n_splits: int = 5,
    random_state: int = DEFAULT_RANDOM_SEED,
) -> Optional[dict]:
    """
    Run the complete gated classification workflow.

    Returns None when the target is not scientifically ready.
    """

    supervised_df = (
        assemble_supervised_dataset(
            feature_df=feature_df,
            target_df=target_df,
            target_column=
                target_column,
        )
    )

    (
        configuration_ids,
        X,
        y,
    ) = split_features_and_target(
        supervised_df=
            supervised_df,
        target_column=
            target_column,
    )

    readiness = (
        check_classification_readiness(
            y=y,
            minimum_samples=
                minimum_samples,
            minimum_samples_per_class=
                minimum_samples_per_class,
        )
    )

    if not readiness[
        "is_ready"
    ]:

        print(
            "Classification training skipped."
        )

        print(
            f"Available labels: "
            f"{readiness['available_samples']}"
        )

        print(
            f"Minimum required: "
            f"{readiness['minimum_samples']}"
        )

        return None

    comparison_df = (
        compare_classification_models(
            models=models,
            X=X,
            y=y,
            n_splits=n_splits,
            random_state=random_state,
        )
    )

    return {
        "target":
            target_column,

        "configuration_ids":
            configuration_ids,

        "X":
            X,

        "y":
            y,

        "readiness":
            readiness,

        "model_comparison":
            comparison_df,
    }
