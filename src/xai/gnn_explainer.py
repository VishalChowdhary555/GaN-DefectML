"""
Graph neural network explainability utilities for GaN-DefectML.

This module provides diagnostic and supervised GNN explanation tools:

- node masking,
- edge masking,
- graph-descriptor masking,
- individual node importance,
- chemically grouped edge-pair importance,
- graph-descriptor importance,
- defect-site localization,
- and complete graph explanation reports.

Important
---------
Mask-based explanations measure model sensitivity to perturbations.
For an untrained diagnostic model they verify that information channels
are connected to the output, but they must NOT be interpreted as
scientific defect-property explanations.

Scientific interpretation should only be performed after training on
validated target labels.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import torch


DEFAULT_MASK_BASELINE = "zero"


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def _get_model_device(
    model,
) -> torch.device:
    """
    Return the device containing the model parameters.
    """

    try:
        return next(
            model.parameters()
        ).device

    except StopIteration:
        return torch.device(
            "cpu"
        )


def _clone_graph_to_device(
    graph,
    device: torch.device,
):
    """
    Clone a PyG graph and move it to the requested device.
    """

    return graph.clone().to(
        device
    )


@torch.no_grad()
def predict_graph(
    model,
    graph,
    device: Optional[
        torch.device
    ] = None,
) -> np.ndarray:
    """
    Run inference on a graph or graph batch.

    Returns
    -------
    numpy.ndarray
        Flattened model output.
    """

    if device is None:
        device = _get_model_device(
            model
        )

    model = model.to(
        device
    )

    model.eval()

    graph = (
        _clone_graph_to_device(
            graph,
            device,
        )
    )

    prediction = (
        model(
            graph
        )
        .reshape(
            -1
        )
    )

    return (
        prediction
        .detach()
        .cpu()
        .numpy()
    )


def calculate_output_change(
    original_prediction,
    perturbed_prediction,
) -> np.ndarray:
    """
    Calculate absolute prediction change after perturbation.
    """

    original_prediction = (
        np.asarray(
            original_prediction,
            dtype=float,
        )
    )

    perturbed_prediction = (
        np.asarray(
            perturbed_prediction,
            dtype=float,
        )
    )

    if (
        original_prediction.shape
        != perturbed_prediction.shape
    ):

        raise ValueError(
            "Original and perturbed predictions "
            "must have identical shapes."
        )

    return np.abs(
        original_prediction
        - perturbed_prediction
    )


# ---------------------------------------------------------------------
# Baseline generation
# ---------------------------------------------------------------------

def _baseline_tensor(
    tensor: torch.Tensor,
    baseline: str = DEFAULT_MASK_BASELINE,
) -> torch.Tensor:
    """
    Construct a masking baseline for a feature tensor.

    Supported baselines
    -------------------
    zero
        Replace selected information with zero.

    mean
        Replace selected information with the mean feature vector.
    """

    baseline = (
        baseline
        .strip()
        .lower()
    )

    if baseline == "zero":

        return torch.zeros_like(
            tensor
        )

    if baseline == "mean":

        if tensor.ndim == 1:

            mean_value = (
                tensor.mean()
            )

            return torch.full_like(
                tensor,
                mean_value,
            )

        mean_vector = tensor.mean(
            dim=0,
            keepdim=True,
        )

        return (
            mean_vector
            .expand_as(
                tensor
            )
            .clone()
        )

    raise ValueError(
        "baseline must be either "
        "'zero' or 'mean'."
    )


# ---------------------------------------------------------------------
# Whole-channel masking
# ---------------------------------------------------------------------

@torch.no_grad()
def calculate_node_channel_sensitivity(
    model,
    graph,
    device: Optional[
        torch.device
    ] = None,
    baseline: str = "zero",
) -> dict:
    """
    Measure sensitivity to the complete node-feature channel.
    """

    if device is None:
        device = _get_model_device(
            model
        )

    original_graph = (
        _clone_graph_to_device(
            graph,
            device,
        )
    )

    masked_graph = (
        _clone_graph_to_device(
            graph,
            device,
        )
    )

    original_prediction = (
        predict_graph(
            model,
            original_graph,
            device,
        )
    )

    masked_graph.x = (
        _baseline_tensor(
            masked_graph.x,
            baseline,
        )
    )

    masked_prediction = (
        predict_graph(
            model,
            masked_graph,
            device,
        )
    )

    change = (
        calculate_output_change(
            original_prediction,
            masked_prediction,
        )
    )

    return {
        "channel":
            "node_features",

        "mean_absolute_output_change":
            float(
                change.mean()
            ),

        "maximum_output_change":
            float(
                change.max()
            ),

        "influences_output":
            bool(
                np.any(
                    change > 1e-8
                )
            ),
    }


@torch.no_grad()
def calculate_edge_channel_sensitivity(
    model,
    graph,
    device: Optional[
        torch.device
    ] = None,
    baseline: str = "zero",
) -> dict:
    """
    Measure sensitivity to the complete edge-feature channel.
    """

    if device is None:
        device = _get_model_device(
            model
        )

    original_graph = (
        _clone_graph_to_device(
            graph,
            device,
        )
    )

    masked_graph = (
        _clone_graph_to_device(
            graph,
            device,
        )
    )

    original_prediction = (
        predict_graph(
            model,
            original_graph,
            device,
        )
    )

    masked_graph.edge_attr = (
        _baseline_tensor(
            masked_graph.edge_attr,
            baseline,
        )
    )

    masked_prediction = (
        predict_graph(
            model,
            masked_graph,
            device,
        )
    )

    change = (
        calculate_output_change(
            original_prediction,
            masked_prediction,
        )
    )

    return {
        "channel":
            "edge_features",

        "mean_absolute_output_change":
            float(
                change.mean()
            ),

        "maximum_output_change":
            float(
                change.max()
            ),

        "influences_output":
            bool(
                np.any(
                    change > 1e-8
                )
            ),
    }


@torch.no_grad()
def calculate_graph_attribute_sensitivity(
    model,
    graph,
    device: Optional[
        torch.device
    ] = None,
    baseline: str = "zero",
) -> dict:
    """
    Measure sensitivity to the graph-level descriptor channel.
    """

    if not hasattr(
        graph,
        "graph_attr",
    ):

        raise AttributeError(
            "Graph does not contain graph_attr."
        )

    if device is None:
        device = _get_model_device(
            model
        )

    original_graph = (
        _clone_graph_to_device(
            graph,
            device,
        )
    )

    masked_graph = (
        _clone_graph_to_device(
            graph,
            device,
        )
    )

    original_prediction = (
        predict_graph(
            model,
            original_graph,
            device,
        )
    )

    masked_graph.graph_attr = (
        _baseline_tensor(
            masked_graph.graph_attr,
            baseline,
        )
    )

    masked_prediction = (
        predict_graph(
            model,
            masked_graph,
            device,
        )
    )

    change = (
        calculate_output_change(
            original_prediction,
            masked_prediction,
        )
    )

    return {
        "channel":
            "graph_attributes",

        "mean_absolute_output_change":
            float(
                change.mean()
            ),

        "maximum_output_change":
            float(
                change.max()
            ),

        "influences_output":
            bool(
                np.any(
                    change > 1e-8
                )
            ),
    }


def calculate_information_channel_sensitivity(
    model,
    graph,
    device: Optional[
        torch.device
    ] = None,
    baseline: str = "zero",
) -> pd.DataFrame:
    """
    Compare node, edge, and global descriptor information channels.
    """

    records = [
        calculate_node_channel_sensitivity(
            model=model,
            graph=graph,
            device=device,
            baseline=baseline,
        ),

        calculate_edge_channel_sensitivity(
            model=model,
            graph=graph,
            device=device,
            baseline=baseline,
        ),

        calculate_graph_attribute_sensitivity(
            model=model,
            graph=graph,
            device=device,
            baseline=baseline,
        ),
    ]

    return (
        pd.DataFrame(
            records
        )
        .sort_values(
            "mean_absolute_output_change",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


# ---------------------------------------------------------------------
# Node importance
# ---------------------------------------------------------------------

@torch.no_grad()
def calculate_node_mask_importance(
    model,
    graph,
    device: Optional[
        torch.device
    ] = None,
    baseline: str = "zero",
    element_symbols: Optional[
        Iterable[str]
    ] = None,
) -> pd.DataFrame:
    """
    Calculate node-level importance by masking one atomic feature
    vector at a time.

    Notes
    -----
    The graph topology itself is retained. Only the selected node's
    feature vector is replaced.
    """

    if device is None:
        device = _get_model_device(
            model
        )

    original_prediction = (
        predict_graph(
            model,
            graph,
            device,
        )
    )

    number_of_nodes = int(
        graph.num_nodes
    )

    if element_symbols is not None:

        element_symbols = list(
            element_symbols
        )

        if len(
            element_symbols
        ) != number_of_nodes:

            raise ValueError(
                "element_symbols length must "
                "equal the number of graph nodes."
            )

    records = []

    for node_index in range(
        number_of_nodes
    ):

        masked_graph = (
            _clone_graph_to_device(
                graph,
                device,
            )
        )

        baseline_matrix = (
            _baseline_tensor(
                masked_graph.x,
                baseline,
            )
        )

        masked_graph.x[
            node_index
        ] = baseline_matrix[
            node_index
        ]

        masked_prediction = (
            predict_graph(
                model,
                masked_graph,
                device,
            )
        )

        change = (
            calculate_output_change(
                original_prediction,
                masked_prediction,
            )
        )

        record = {
            "node_index":
                int(
                    node_index
                ),

            "importance":
                float(
                    change.mean()
                ),
        }

        if element_symbols is not None:

            record[
                "element"
            ] = element_symbols[
                node_index
            ]

        records.append(
            record
        )

    return (
        pd.DataFrame(
            records
        )
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


# ---------------------------------------------------------------------
# Edge-pair importance
# ---------------------------------------------------------------------

def _infer_element_symbols_from_structure(
    structure,
) -> list[str]:
    """
    Extract ordered atomic symbols from a Pymatgen Structure.
    """

    return [
        site.specie.symbol
        for site in structure
    ]


def _canonical_element_pair(
    element_a: str,
    element_b: str,
) -> str:
    """
    Construct an order-independent chemical pair label.
    """

    return "-".join(
        sorted(
            [
                str(
                    element_a
                ),
                str(
                    element_b
                ),
            ]
        )
    )


@torch.no_grad()
def calculate_edge_pair_mask_importance(
    model,
    graph,
    structure=None,
    element_symbols: Optional[
        Iterable[str]
    ] = None,
    device: Optional[
        torch.device
    ] = None,
    baseline: str = "zero",
) -> pd.DataFrame:
    """
    Calculate chemically grouped edge importance.

    All edges belonging to a chemical pair such as Ga-N, Ga-Ga,
    Mg-N, etc. are masked simultaneously.

    This is the reusable implementation of the edge-pair diagnostic
    used during notebook XAI development.
    """

    if device is None:
        device = _get_model_device(
            model
        )

    if element_symbols is None:

        if structure is None:

            raise ValueError(
                "Either structure or element_symbols "
                "must be supplied."
            )

        element_symbols = (
            _infer_element_symbols_from_structure(
                structure
            )
        )

    element_symbols = list(
        element_symbols
    )

    if len(
        element_symbols
    ) != int(
        graph.num_nodes
    ):

        raise ValueError(
            "Element-symbol count does not "
            "match graph node count."
        )

    edge_index = (
        graph.edge_index
        .detach()
        .cpu()
        .numpy()
    )

    edge_pair_labels = []

    for edge_position in range(
        edge_index.shape[1]
    ):

        source = int(
            edge_index[
                0,
                edge_position,
            ]
        )

        target = int(
            edge_index[
                1,
                edge_position,
            ]
        )

        pair = (
            _canonical_element_pair(
                element_symbols[
                    source
                ],
                element_symbols[
                    target
                ],
            )
        )

        edge_pair_labels.append(
            pair
        )

    edge_pair_labels = np.asarray(
        edge_pair_labels,
        dtype=object,
    )

    unique_pairs = sorted(
        set(
            edge_pair_labels.tolist()
        )
    )

    original_prediction = (
        predict_graph(
            model,
            graph,
            device,
        )
    )

    records = []

    for pair in unique_pairs:

        pair_mask = (
            edge_pair_labels
            == pair
        )

        edge_positions = np.where(
            pair_mask
        )[0]

        masked_graph = (
            _clone_graph_to_device(
                graph,
                device,
            )
        )

        baseline_edges = (
            _baseline_tensor(
                masked_graph.edge_attr,
                baseline,
            )
        )

        edge_position_tensor = (
            torch.as_tensor(
                edge_positions,
                dtype=torch.long,
                device=device,
            )
        )

        masked_graph.edge_attr[
            edge_position_tensor
        ] = baseline_edges[
            edge_position_tensor
        ]

        masked_prediction = (
            predict_graph(
                model,
                masked_graph,
                device,
            )
        )

        change = (
            calculate_output_change(
                original_prediction,
                masked_prediction,
            )
        )

        records.append(
            {
                "edge_pair":
                    pair,

                "number_of_directed_edges":
                    int(
                        len(
                            edge_positions
                        )
                    ),

                "importance":
                    float(
                        change.mean()
                    ),

                "maximum_output_change":
                    float(
                        change.max()
                    ),
            }
        )

    return (
        pd.DataFrame(
            records
        )
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


# ---------------------------------------------------------------------
# Individual graph-descriptor importance
# ---------------------------------------------------------------------

@torch.no_grad()
def calculate_graph_descriptor_importance(
    model,
    graph,
    feature_names: Iterable[str],
    device: Optional[
        torch.device
    ] = None,
    baseline: str = "zero",
) -> pd.DataFrame:
    """
    Mask individual graph-level descriptors and measure prediction
    sensitivity.
    """

    if not hasattr(
        graph,
        "graph_attr",
    ):

        raise AttributeError(
            "Graph does not contain graph_attr."
        )

    feature_names = list(
        feature_names
    )

    graph_attr = graph.graph_attr

    if graph_attr.ndim == 1:

        feature_dimension = (
            graph_attr.shape[0]
        )

    else:

        feature_dimension = (
            graph_attr.shape[-1]
        )

    if len(
        feature_names
    ) != feature_dimension:

        raise ValueError(
            "feature_names length does not "
            "match graph_attr dimension."
        )

    if device is None:
        device = _get_model_device(
            model
        )

    original_prediction = (
        predict_graph(
            model,
            graph,
            device,
        )
    )

    records = []

    for feature_index, feature_name in enumerate(
        feature_names
    ):

        masked_graph = (
            _clone_graph_to_device(
                graph,
                device,
            )
        )

        baseline_attributes = (
            _baseline_tensor(
                masked_graph.graph_attr,
                baseline,
            )
        )

        if (
            masked_graph.graph_attr.ndim
            == 1
        ):

            masked_graph.graph_attr[
                feature_index
            ] = baseline_attributes[
                feature_index
            ]

        else:

            masked_graph.graph_attr[
                :,
                feature_index
            ] = baseline_attributes[
                :,
                feature_index
            ]

        masked_prediction = (
            predict_graph(
                model,
                masked_graph,
                device,
            )
        )

        change = (
            calculate_output_change(
                original_prediction,
                masked_prediction,
            )
        )

        records.append(
            {
                "feature":
                    feature_name,

                "feature_index":
                    int(
                        feature_index
                    ),

                "importance":
                    float(
                        change.mean()
                    ),

                "maximum_output_change":
                    float(
                        change.max()
                    ),
            }
        )

    return (
        pd.DataFrame(
            records
        )
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


# ---------------------------------------------------------------------
# Defect-site helpers
# ---------------------------------------------------------------------

def identify_interstitial_defect_atom(
    pristine_structure,
    defect_structure,
) -> Optional[int]:
    """
    Identify the inserted atom in a single-interstitial structure.

    The generated GaN-DefectML structure library appends inserted
    interstitial atoms to the end of the Pymatgen Structure.
    """

    if (
        len(
            defect_structure
        )
        !=
        len(
            pristine_structure
        )
        + 1
    ):

        return None

    return int(
        len(
            defect_structure
        )
        - 1
    )


def identify_substitution_defect_atom(
    pristine_structure,
    defect_structure,
) -> Optional[int]:
    """
    Identify the changed atomic site for a substitution or antisite.
    """

    if len(
        pristine_structure
    ) != len(
        defect_structure
    ):

        return None

    changed_indices = []

    for index, (
        pristine_site,
        defect_site,
    ) in enumerate(
        zip(
            pristine_structure,
            defect_structure,
        )
    ):

        if (
            pristine_site.specie.symbol
            !=
            defect_site.specie.symbol
        ):

            changed_indices.append(
                index
            )

    if len(
        changed_indices
    ) != 1:

        return None

    return int(
        changed_indices[0]
    )


def get_defect_atom_report(
    defect_structure,
    defect_atom_index: int,
) -> dict:
    """
    Generate coordinates and species information for a defect atom.
    """

    if not (
        0
        <= defect_atom_index
        < len(
            defect_structure
        )
    ):

        raise IndexError(
            "Defect atom index is outside "
            "the structure."
        )

    site = defect_structure[
        defect_atom_index
    ]

    return {
        "defect_atom_index":
            int(
                defect_atom_index
            ),

        "element":
            site.specie.symbol,

        "fractional_coordinates":
            np.asarray(
                site.frac_coords,
                dtype=float,
            ),

        "cartesian_coordinates_A":
            np.asarray(
                site.coords,
                dtype=float,
            ),
    }


# ---------------------------------------------------------------------
# Defect-centric node explanation
# ---------------------------------------------------------------------

def calculate_defect_node_rank(
    node_importance_df: pd.DataFrame,
    defect_atom_index: int,
) -> dict:
    """
    Determine where the known defect atom ranks in node importance.
    """

    required_columns = {
        "node_index",
        "importance",
    }

    missing = (
        required_columns
        - set(
            node_importance_df.columns
        )
    )

    if missing:

        raise ValueError(
            "Node importance table is missing "
            f"columns: {sorted(missing)}"
        )

    ranked = (
        node_importance_df
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    matches = ranked.index[
        ranked[
            "node_index"
        ]
        == defect_atom_index
    ].tolist()

    if not matches:

        return {
            "defect_atom_found":
                False,

            "defect_atom_rank":
                None,

            "defect_atom_importance":
                np.nan,
        }

    row_index = matches[0]

    return {
        "defect_atom_found":
            True,

        "defect_atom_rank":
            int(
                row_index + 1
            ),

        "defect_atom_importance":
            float(
                ranked.loc[
                    row_index,
                    "importance",
                ]
            ),
    }


# ---------------------------------------------------------------------
# Complete diagnostic explanation
# ---------------------------------------------------------------------

def explain_graph_diagnostic(
    model,
    graph,
    structure,
    global_feature_names: Iterable[str],
    device: Optional[
        torch.device
    ] = None,
    baseline: str = "zero",
    defect_atom_index: Optional[
        int
    ] = None,
) -> Dict[str, object]:
    """
    Generate a complete diagnostic explanation for one graph.

    This function is suitable for testing an untrained architecture,
    but results from such a model should only be interpreted as
    connectivity/sensitivity diagnostics.
    """

    element_symbols = (
        _infer_element_symbols_from_structure(
            structure
        )
    )

    original_prediction = (
        predict_graph(
            model=model,
            graph=graph,
            device=device,
        )
    )

    channel_importance_df = (
        calculate_information_channel_sensitivity(
            model=model,
            graph=graph,
            device=device,
            baseline=baseline,
        )
    )

    node_importance_df = (
        calculate_node_mask_importance(
            model=model,
            graph=graph,
            device=device,
            baseline=baseline,
            element_symbols=
                element_symbols,
        )
    )

    edge_pair_importance_df = (
        calculate_edge_pair_mask_importance(
            model=model,
            graph=graph,
            structure=structure,
            device=device,
            baseline=baseline,
        )
    )

    graph_descriptor_importance_df = (
        calculate_graph_descriptor_importance(
            model=model,
            graph=graph,
            feature_names=
                global_feature_names,
            device=device,
            baseline=baseline,
        )
    )

    defect_rank = None

    if defect_atom_index is not None:

        defect_rank = (
            calculate_defect_node_rank(
                node_importance_df=
                    node_importance_df,
                defect_atom_index=
                    defect_atom_index,
            )
        )

    return {
        "configuration_id":
            getattr(
                graph,
                "configuration_id",
                None,
            ),

        "original_prediction":
            original_prediction,

        "channel_importance":
            channel_importance_df,

        "node_importance":
            node_importance_df,

        "edge_pair_importance":
            edge_pair_importance_df,

        "graph_descriptor_importance":
            graph_descriptor_importance_df,

        "defect_node_rank":
            defect_rank,

        "diagnostic_only":
            True,
    }


# ---------------------------------------------------------------------
# Scientific interpretation gate
# ---------------------------------------------------------------------

def explanation_interpretation_status(
    model_is_trained: bool,
    validated_labels_available: bool,
) -> dict:
    """
    Explicitly distinguish architecture diagnostics from scientific XAI.
    """

    scientific_interpretation_allowed = bool(
        model_is_trained
        and validated_labels_available
    )

    if scientific_interpretation_allowed:

        message = (
            "The model is trained on validated labels. "
            "Explanation results may be analyzed scientifically, "
            "subject to normal model-validation limitations."
        )

    else:

        message = (
            "Explanation results are diagnostic only. "
            "They verify model sensitivity and information flow, "
            "but must not be interpreted as learned defect physics."
        )

    return {
        "scientific_interpretation_allowed":
            scientific_interpretation_allowed,

        "message":
            message,
    }
