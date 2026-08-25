"""
Conventional supervised-learning utilities for GaN-DefectML.
"""

from .target_validation import (
    CLASSIFICATION_TARGETS,
    REGRESSION_TARGETS,
    TARGET_SCHEMA_COLUMNS,
    audit_target_completeness,
    build_configuration_target_table,
    build_target_readiness_table,
    create_blank_target_template,
    validate_target_table,
)

from .training import (
    assemble_supervised_dataset,
    check_classification_readiness,
    check_regression_readiness,
    compare_classification_models,
    compare_regression_models,
    fit_final_model,
    run_classification_training,
    run_regression_training,
)

from .evaluation import (
    build_evaluation_report,
    build_regression_prediction_table,
    build_classification_prediction_table,
    rank_classification_models,
    rank_regression_models,
    regression_metrics,
    classification_metrics,
)

__all__ = [
    "TARGET_SCHEMA_COLUMNS",
    "REGRESSION_TARGETS",
    "CLASSIFICATION_TARGETS",
    "create_blank_target_template",
    "validate_target_table",
    "audit_target_completeness",
    "build_target_readiness_table",
    "build_configuration_target_table",
    "assemble_supervised_dataset",
    "check_regression_readiness",
    "check_classification_readiness",
    "compare_regression_models",
    "compare_classification_models",
    "fit_final_model",
    "run_regression_training",
    "run_classification_training",
    "regression_metrics",
    "classification_metrics",
    "build_regression_prediction_table",
    "build_classification_prediction_table",
    "build_evaluation_report",
    "rank_regression_models",
    "rank_classification_models",
]
