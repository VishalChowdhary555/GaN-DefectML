"""
Classical machine-learning model definitions for GaN-DefectML.

The models defined here operate on the compact physics-informed
tabular feature representation.

Supported tasks
---------------
Regression:
    - Ridge Regression
    - Random Forest Regression
    - Extra Trees Regression
    - Gradient Boosting Regression

Classification:
    - Logistic Regression
    - Random Forest Classification
    - Extra Trees Classification
    - Gradient Boosting Classification

No model is fitted automatically. Training is handled separately after
validated supervised labels pass the scientific readiness checks.
"""

from __future__ import annotations

from typing import Dict

from sklearn.base import BaseEstimator

from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)

from sklearn.linear_model import (
    LogisticRegression,
    Ridge,
)

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler


DEFAULT_RANDOM_SEED = 42


# ---------------------------------------------------------------------
# Regression models
# ---------------------------------------------------------------------

def build_ridge_regressor(
    alpha: float = 1.0,
) -> Pipeline:
    """
    Construct a standardized Ridge regression pipeline.

    Parameters
    ----------
    alpha : float, default=1.0
        L2 regularization strength.

    Returns
    -------
    sklearn.pipeline.Pipeline
        StandardScaler followed by Ridge regression.
    """

    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                Ridge(
                    alpha=alpha,
                ),
            ),
        ]
    )


def build_random_forest_regressor(
    random_state: int = DEFAULT_RANDOM_SEED,
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
) -> RandomForestRegressor:
    """
    Construct a Random Forest regression model.
    """

    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )


def build_extra_trees_regressor(
    random_state: int = DEFAULT_RANDOM_SEED,
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
) -> ExtraTreesRegressor:
    """
    Construct an Extra Trees regression model.
    """

    return ExtraTreesRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )


def build_gradient_boosting_regressor(
    random_state: int = DEFAULT_RANDOM_SEED,
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    max_depth: int = 3,
) -> GradientBoostingRegressor:
    """
    Construct a Gradient Boosting regression model.
    """

    return GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
    )


def get_regression_models(
    random_state: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, BaseEstimator]:
    """
    Return the default regression-model collection.

    Returns
    -------
    dict
        Model name -> unfitted estimator.
    """

    return {
        "Ridge":
            build_ridge_regressor(),

        "RandomForest":
            build_random_forest_regressor(
                random_state=random_state,
            ),

        "ExtraTrees":
            build_extra_trees_regressor(
                random_state=random_state,
            ),

        "GradientBoosting":
            build_gradient_boosting_regressor(
                random_state=random_state,
            ),
    }


# ---------------------------------------------------------------------
# Classification models
# ---------------------------------------------------------------------

def build_logistic_classifier(
    random_state: int = DEFAULT_RANDOM_SEED,
    max_iter: int = 2000,
) -> Pipeline:
    """
    Construct a standardized Logistic Regression classifier.
    """

    return Pipeline(
        steps=[
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
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
) -> RandomForestClassifier:
    """
    Construct a Random Forest classifier.
    """

    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


def build_extra_trees_classifier(
    random_state: int = DEFAULT_RANDOM_SEED,
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
) -> ExtraTreesClassifier:
    """
    Construct an Extra Trees classifier.
    """

    return ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


def build_gradient_boosting_classifier(
    random_state: int = DEFAULT_RANDOM_SEED,
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    max_depth: int = 3,
) -> GradientBoostingClassifier:
    """
    Construct a Gradient Boosting classifier.
    """

    return GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
    )


def get_classification_models(
    random_state: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, BaseEstimator]:
    """
    Return the default classification-model collection.

    Returns
    -------
    dict
        Model name -> unfitted estimator.
    """

    return {
        "LogisticRegression":
            build_logistic_classifier(
                random_state=random_state,
            ),

        "RandomForest":
            build_random_forest_classifier(
                random_state=random_state,
            ),

        "ExtraTrees":
            build_extra_trees_classifier(
                random_state=random_state,
            ),

        "GradientBoosting":
            build_gradient_boosting_classifier(
                random_state=random_state,
            ),
    }


# ---------------------------------------------------------------------
# Unified model factory
# ---------------------------------------------------------------------

def get_model_collection(
    problem_type: str,
    random_state: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, BaseEstimator]:
    """
    Return models appropriate for a supervised-learning task.

    Parameters
    ----------
    problem_type : str
        Either ``"regression"`` or ``"classification"``.

    random_state : int
        Random seed used by stochastic estimators.

    Returns
    -------
    dict
        Model-name -> estimator mapping.
    """

    problem_type = (
        problem_type
        .strip()
        .lower()
    )

    if problem_type == "regression":

        return get_regression_models(
            random_state=random_state,
        )

    if problem_type == "classification":

        return get_classification_models(
            random_state=random_state,
        )

    raise ValueError(
        "problem_type must be either "
        "'regression' or 'classification'."
    )


# ---------------------------------------------------------------------
# Model inspection
# ---------------------------------------------------------------------

def summarize_model_collection(
    models: Dict[str, BaseEstimator],
):
    """
    Print a concise summary of an unfitted model collection.
    """

    print(
        "=" * 70
    )

    print(
        "CLASSICAL MODEL COLLECTION"
    )

    print(
        "=" * 70
    )

    print(
        f"Number of models: {len(models)}"
    )

    print()

    for model_name, model in models.items():

        print(
            f"{model_name:<22} | "
            f"{model.__class__.__name__}"
        )
