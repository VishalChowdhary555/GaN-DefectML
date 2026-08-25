"""
Model architectures for GaN-DefectML.
"""

from .classical_models import (
    get_classification_models,
    get_model_collection,
    get_regression_models,
)

from .gnn_model import (
    EdgeAwareGaNDefectGNN,
    GaNDefectGNN,
    build_baseline_gnn,
    build_default_gnn,
    build_edge_aware_gnn,
    count_trainable_parameters,
)

__all__ = [
    "get_regression_models",
    "get_classification_models",
    "get_model_collection",
    "GaNDefectGNN",
    "EdgeAwareGaNDefectGNN",
    "build_baseline_gnn",
    "build_edge_aware_gnn",
    "build_default_gnn",
    "count_trainable_parameters",
]
