"""
Shared utility functions for GaN-DefectML.

This package centralizes:
- project configuration,
- reproducibility,
- filesystem paths,
- data and model artifact I/O,
- and project-wide validation.
"""

# =====================================================================
# Configuration
# =====================================================================

from .config import (
    CONFIG_DIR,
    DATA_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    MODEL_DIR,
    PREDICTIONS_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    RESULTS_DIR,
    TARGET_DATA_DIR,
    GraphConfig,
    LearningConfig,
    ProjectConfig,
    XAIConfig,
    configure_deterministic_torch,
    create_project_directories,
    get_default_config,
    get_device,
    get_device_summary,
    initialize_project,
    load_json_config,
    save_config,
    save_learning_config,
    save_xai_config,
    set_global_seed,
)


# =====================================================================
# Input / output
# =====================================================================

from .io import (
    build_file_manifest,
    ensure_directory,
    ensure_parent_directory,
    load_dataframe,
    load_graph_dataset,
    load_json,
    load_numpy_array,
    load_structure,
    load_structure_library,
    load_target_table,
    load_torch_checkpoint,
    require_file,
    save_dataframe,
    save_feature_matrix,
    save_graph_dataset,
    save_json,
    save_metrics,
    save_numpy_array,
    save_predictions,
    save_structure,
    save_structure_library,
    save_target_table,
    save_torch_checkpoint,
)


# =====================================================================
# Validation
# =====================================================================

from .validation import (
    validate_expected_configurations,
    validate_feature_matrix,
    validate_finite_dataframe,
    validate_graph_dataset_ids,
    validate_graph_structure_alignment,
    validate_graph_tensor_finiteness,
    validate_no_constant_features,
    validate_pipeline_alignment,
    validate_structure_composition,
    validate_structure_library,
    validate_target_alignment,
    validate_unique_configuration_ids,
)


# =====================================================================
# Public API
# =====================================================================

__all__ = [
    # Configuration
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "TARGET_DATA_DIR",
    "MODEL_DIR",
    "RESULTS_DIR",
    "FIGURES_DIR",
    "METRICS_DIR",
    "PREDICTIONS_DIR",
    "CONFIG_DIR",
    "GraphConfig",
    "LearningConfig",
    "XAIConfig",
    "ProjectConfig",
    "get_default_config",
    "create_project_directories",
    "set_global_seed",
    "configure_deterministic_torch",
    "get_device",
    "get_device_summary",
    "save_config",
    "save_learning_config",
    "save_xai_config",
    "load_json_config",
    "initialize_project",

    # I/O
    "ensure_parent_directory",
    "ensure_directory",
    "require_file",
    "save_json",
    "load_json",
    "save_dataframe",
    "load_dataframe",
    "save_numpy_array",
    "load_numpy_array",
    "save_structure",
    "load_structure",
    "save_structure_library",
    "load_structure_library",
    "save_feature_matrix",
    "save_target_table",
    "load_target_table",
    "save_graph_dataset",
    "load_graph_dataset",
    "save_torch_checkpoint",
    "load_torch_checkpoint",
    "save_metrics",
    "save_predictions",
    "build_file_manifest",

    # Validation
    "validate_unique_configuration_ids",
    "validate_expected_configurations",
    "validate_finite_dataframe",
    "validate_no_constant_features",
    "validate_structure_library",
    "validate_structure_composition",
    "validate_feature_matrix",
    "validate_target_alignment",
    "validate_graph_tensor_finiteness",
    "validate_graph_structure_alignment",
    "validate_graph_dataset_ids",
    "validate_pipeline_alignment",
]
