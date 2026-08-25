"""
Classical machine-learning model definitions for GaN-DefectML.

This module defines baseline estimators for tabular defect-property
prediction using the compact physics-informed descriptor matrix.

Training and evaluation logic is intentionally kept separate under
``src/ml``.
"""

from __future__ import annotations

from typing import Dict

from sklearn.base import BaseEstimator

from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)

from sklearn.impute import SimpleImputer

from sklearn.linear_model import (
    LogisticRegression,
    Ridge,
)

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler


DEFAULT_RANDOM_SEED = 42


# =====================================================================
# Regression
# =====================================================================

def build_ridge_regressor(
    alpha: float = 1.0,
) -> Pipeline:
    """
    Build a regularized linear regression baseline.
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                Ridge(
                    alpha=alpha
                ),
            ),
        ]
    )


def build_random_forest_regressor(
    random_state: int = DEFAULT_RANDOM_SEED,
    n_estimators: int = 500,
    min_samples_leaf: int = 2,
    max_depth: int | None = None,
) -> Pipeline:
    """
    Build a Random Forest regression model.
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    min_samples_leaf=min_samples_leaf,
                    max_depth=max_depth,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_extra_trees_regressor(
    random_state: int = DEFAULT_RANDOM_SEED,
    n_estimators: int = 500,
    min_samples_leaf: int = 2,
    max_depth: int | None = None,
) -> Pipeline:
    """
    Build an Extra Trees regression model.
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=n_estimators,
                    min_samples_leaf=min_samples_leaf,
                    max_depth=max_depth,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_hist_gradient_boosting_regressor(
    random_state: int = DEFAULT_RANDOM_SEED,
    learning_rate: float = 0.05,
    max_iter: int = 300,
    l2_regularization: float = 1.0,
) -> Pipeline:
    """
    Build a histogram gradient-boosting regression model.
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=learning_rate,
                    max_iter=max_iter,
                    l2_regularization=l2_regularization,
                    random_state=random_state,
                ),
            ),
        ]
    )


def get_regression_models(
    random_state: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, BaseEstimator]:
    """
    Return the default regression model collection.
    """

    return {
        "Ridge":
            build_ridge_regressor(),

        "RandomForest":
            build_random_forest_regressor(
                random_state=random_state
            ),

        "ExtraTrees":
            build_extra_trees_regressor(
                random_state=random_state
            ),

        "HistGradientBoosting":
            build_hist_gradient_boosting_regressor(
                random_state=random_state
            ),
    }


# =====================================================================
# Classification
# =====================================================================

def build_logistic_classifier(
    random_state: int = DEFAULT_RANDOM_SEED,
    max_iter: int = 2000,
) -> Pipeline:
    """
    Build a balanced Logistic Regression classifier.
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=max_iter,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_random_forest_classifier(
    random_state: int = DEFAULT_RANDOM_SEED,
    n_estimators: int = 500,
    min_samples_leaf: int = 2,
    max_depth: int | None = None,
) -> Pipeline:
    """
    Build a Random Forest classifier.
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    min_samples_leaf=min_samples_leaf,
                    max_depth=max_depth,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_extra_trees_classifier(
    random_state: int = DEFAULT_RANDOM_SEED,
    n_estimators: int = 500,
    min_samples_leaf: int = 2,
    max_depth: int | None = None,
) -> Pipeline:
    """
    Build an Extra Trees classifier.
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=n_estimators,
                    min_samples_leaf=min_samples_leaf,
                    max_depth=max_depth,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_hist_gradient_boosting_classifier(
    random_state: int = DEFAULT_RANDOM_SEED,
    learning_rate: float = 0.05,
    max_iter: int = 300,
    l2_regularization: float = 1.0,
) -> Pipeline:
    """
    Build a histogram gradient-boosting classifier.
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=learning_rate,
                    max_iter=max_iter,
                    l2_regularization=l2_regularization,
                    random_state=random_state,
                ),
            ),
        ]
    )


def get_classification_models(
    random_state: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, BaseEstimator]:
    """
    Return the default classification model collection.
    """

    return {
        "LogisticRegression":
            build_logistic_classifier(
                random_state=random_state
            ),

        "RandomForest":
            build_random_forest_classifier(
                random_state=random_state
            ),

        "ExtraTrees":
            build_extra_trees_classifier(
                random_state=random_state
            ),

        "HistGradientBoosting":
            build_hist_gradient_boosting_classifier(
                random_state=random_state
            ),
    }


# =====================================================================
# Unified factory
# =====================================================================

def get_model_collection(
    problem_type: str,
    random_state: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, BaseEstimator]:
    """
    Return regression or classification baselines.
    """

    normalized_problem_type = (
        problem_type
        .strip()
        .lower()
    )

    if normalized_problem_type == "regression":
        return get_regression_models(
            random_state=random_state
        )

    if normalized_problem_type == "classification":
        return get_classification_models(
            random_state=random_state
        )

    raise ValueError(
        "problem_type must be either "
        "'regression' or 'classification'."
    )
