"""
Explainability utilities for tabular and graph-based GaN defect models.
"""

from .tabular_explainer import (
    aggregate_grouped_importance,
    calculate_descriptor_ablation,
    calculate_permutation_importance,
    calculate_shap_values,
    check_tabular_xai_readiness,
    run_tabular_xai,
)

from .gnn_explainer import (
    calculate_edge_pair_mask_importance,
    calculate_graph_descriptor_importance,
    calculate_information_channel_sensitivity,
    calculate_node_mask_importance,
    explain_graph_diagnostic,
    explanation_interpretation_status,
)

from .visualization import (
    plot_edge_pair_importance,
    plot_feature_importance,
    plot_graph_descriptor_importance,
    plot_grouped_importance,
    plot_importance_vs_defect_distance,
    plot_node_importance,
    plot_prediction_scatter,
    plot_shap_importance,
    save_figure,
)

__all__ = [
    "check_tabular_xai_readiness",
    "calculate_permutation_importance",
    "aggregate_grouped_importance",
    "calculate_descriptor_ablation",
    "calculate_shap_values",
    "run_tabular_xai",
    "calculate_node_mask_importance",
    "calculate_edge_pair_mask_importance",
    "calculate_graph_descriptor_importance",
    "calculate_information_channel_sensitivity",
    "explain_graph_diagnostic",
    "explanation_interpretation_status",
    "plot_feature_importance",
    "plot_grouped_importance",
    "plot_shap_importance",
    "plot_node_importance",
    "plot_edge_pair_importance",
    "plot_graph_descriptor_importance",
    "plot_importance_vs_defect_distance",
    "plot_prediction_scatter",
    "save_figure",
]
