"""
Graph construction, diagnostics, and training utilities for GaN-DefectML.
"""

from .node_features import (
    GRAPH_ELEMENT_ORDER,
    NODE_FEATURE_NAMES,
    build_node_feature_vector,
    build_structure_node_matrix,
    fit_node_normalization,
    normalize_structure_node_matrix,
)

from .graph_builder import (
    build_edge_feature_matrix,
    build_graph_dataset,
    build_periodic_edge_list,
    get_edge_feature_names,
    structure_to_pyg_graph,
    summarize_graph,
    validate_graph,
)

from .diagnostics import (
    build_graph_summary_table,
    compare_cutoff_radii,
    diagnose_interstitial_atom,
    summarize_graph_dataset,
    validate_graph_dataset,
)

from .training import (
    assess_gnn_regression_readiness,
    create_graph_loader,
    cross_validate_gnn_regressor,
    evaluate_gnn,
    fit_gnn_regressor,
    get_labeled_graphs,
    run_gnn_regression_pipeline,
)

__all__ = [
    "GRAPH_ELEMENT_ORDER",
    "NODE_FEATURE_NAMES",
    "build_node_feature_vector",
    "build_structure_node_matrix",
    "fit_node_normalization",
    "normalize_structure_node_matrix",
    "build_periodic_edge_list",
    "build_edge_feature_matrix",
    "get_edge_feature_names",
    "structure_to_pyg_graph",
    "build_graph_dataset",
    "validate_graph",
    "summarize_graph",
    "validate_graph_dataset",
    "build_graph_summary_table",
    "summarize_graph_dataset",
    "compare_cutoff_radii",
    "diagnose_interstitial_atom",
    "get_labeled_graphs",
    "assess_gnn_regression_readiness",
    "create_graph_loader",
    "fit_gnn_regressor",
    "evaluate_gnn",
    "cross_validate_gnn_regressor",
    "run_gnn_regression_pipeline",
]
