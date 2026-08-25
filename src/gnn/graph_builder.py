"""
Periodic crystal graph construction utilities for GaN-DefectML.

This module provides:
- periodic edge construction from Pymatgen structures,
- Gaussian radial basis expansion of interatomic distances,
- conversion to PyTorch Geometric Data objects,
- graph-level feature attachment,
- graph validation,
- and graph-dataset construction.

The primary graph topology used in this project is a 3.0 Å radial cutoff,
which captures the first-shell tetrahedral Ga-N bonding environment in
wurtzite GaN.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

from torch_geometric.data import Data

from .node_features import (
    normalize_structure_node_matrix,
)


DEFAULT_GRAPH_CUTOFF_A = 3.0
DEFAULT_NUMBER_OF_RBF = 16


def build_periodic_edge_list(
    structure,
    cutoff_radius_A: float = DEFAULT_GRAPH_CUTOFF_A,
):
    """
    Construct directed periodic edges using a radial cutoff.

    Each periodic neighbor relation is represented as a directed edge
    from the central atom to the neighboring atom.

    Parameters
    ----------
    structure
        Pymatgen Structure object.

    cutoff_radius_A : float, default=3.0
        Maximum periodic neighbor distance.

    Returns
    -------
    tuple
        ``(edge_index, edge_distances, edge_vectors)``

        edge_index
            Integer array of shape [2, number_of_edges].

        edge_distances
            Distance array of shape [number_of_edges, 1].

        edge_vectors
            Periodic displacement vectors of shape
            [number_of_edges, 3].
    """

    source_indices = []
    target_indices = []

    edge_distances = []
    edge_vectors = []

    for source_index, source_site in enumerate(
        structure
    ):

        neighbors = structure.get_neighbors(
            source_site,
            r=cutoff_radius_A,
            include_index=True,
            include_image=True,
        )

        for neighbor in neighbors:

            target_index = int(
                neighbor.index
            )

            distance = float(
                neighbor.nn_distance
            )

            if distance <= 1e-8:
                continue

            displacement_vector = (
                np.asarray(
                    neighbor.coords,
                    dtype=float,
                )
                -
                np.asarray(
                    source_site.coords,
                    dtype=float,
                )
            )

            source_indices.append(
                source_index
            )

            target_indices.append(
                target_index
            )

            edge_distances.append(
                [distance]
            )

            edge_vectors.append(
                displacement_vector
            )

    edge_index = np.asarray(
        [
            source_indices,
            target_indices,
        ],
        dtype=np.int64,
    )

    edge_distance_array = np.asarray(
        edge_distances,
        dtype=np.float32,
    )

    edge_vector_array = np.asarray(
        edge_vectors,
        dtype=np.float32,
    )

    if edge_index.ndim != 2:
        raise ValueError(
            "Constructed edge_index must be two-dimensional."
        )

    return (
        edge_index,
        edge_distance_array,
        edge_vector_array,
    )


def create_radial_basis_centers(
    cutoff_radius_A: float = DEFAULT_GRAPH_CUTOFF_A,
    number_of_functions: int = DEFAULT_NUMBER_OF_RBF,
) -> np.ndarray:
    """
    Create evenly spaced Gaussian radial basis centers.
    """

    if number_of_functions < 2:
        raise ValueError(
            "At least two radial basis functions are required."
        )

    return np.linspace(
        0.0,
        cutoff_radius_A,
        number_of_functions,
        dtype=np.float32,
    )


def gaussian_radial_basis(
    distances: np.ndarray,
    cutoff_radius_A: float = DEFAULT_GRAPH_CUTOFF_A,
    number_of_functions: int = DEFAULT_NUMBER_OF_RBF,
) -> np.ndarray:
    """
    Expand edge distances using Gaussian radial basis functions.

    Parameters
    ----------
    distances
        Array of shape [number_of_edges, 1].

    cutoff_radius_A
        Graph cutoff radius.

    number_of_functions
        Number of Gaussian radial basis functions.

    Returns
    -------
    numpy.ndarray
        Expanded edge feature matrix.
    """

    distances = np.asarray(
        distances,
        dtype=np.float32,
    )

    if distances.ndim != 2:
        raise ValueError(
            "Distance array must have shape [N, 1]."
        )

    centers = create_radial_basis_centers(
        cutoff_radius_A=
            cutoff_radius_A,
        number_of_functions=
            number_of_functions,
    )

    spacing = (
        centers[1]
        - centers[0]
    )

    expanded = np.exp(
        -(
            distances
            - centers.reshape(
                1,
                -1,
            )
        ) ** 2
        /
        (
            spacing ** 2
            + 1e-8
        )
    )

    return expanded.astype(
        np.float32
    )


def build_edge_feature_matrix(
    edge_distances: np.ndarray,
    cutoff_radius_A: float = DEFAULT_GRAPH_CUTOFF_A,
    number_of_rbf: int = DEFAULT_NUMBER_OF_RBF,
) -> np.ndarray:
    """
    Build complete edge attributes.

    The first column is raw distance, followed by Gaussian radial
    basis features.

    Returns
    -------
    numpy.ndarray
        Shape [number_of_edges, 1 + number_of_rbf].
    """

    radial_features = gaussian_radial_basis(
        distances=
            edge_distances,
        cutoff_radius_A=
            cutoff_radius_A,
        number_of_functions=
            number_of_rbf,
    )

    edge_features = np.concatenate(
        [
            edge_distances,
            radial_features,
        ],
        axis=1,
    )

    return edge_features.astype(
        np.float32
    )


def get_edge_feature_names(
    number_of_rbf: int = DEFAULT_NUMBER_OF_RBF,
) -> list[str]:
    """
    Return human-readable edge feature names.
    """

    return [
        "raw_distance_A",
        *[
            f"rbf_{index + 1}"
            for index in range(
                number_of_rbf
            )
        ],
    ]


def structure_to_pyg_graph(
    configuration_id: str,
    structure,
    node_normalization_statistics: Dict[str, np.ndarray],
    global_descriptor_vector: np.ndarray,
    cutoff_radius_A: float = DEFAULT_GRAPH_CUTOFF_A,
    number_of_rbf: int = DEFAULT_NUMBER_OF_RBF,
    target_value: Optional[float] = None,
) -> Data:
    """
    Convert one Pymatgen Structure into a PyTorch Geometric graph.

    Parameters
    ----------
    configuration_id
        Structure identifier.

    structure
        Pymatgen Structure object.

    node_normalization_statistics
        Mean/std statistics for continuous node features.

    global_descriptor_vector
        Physics-informed graph-level descriptor vector.

    cutoff_radius_A
        Periodic neighbor cutoff.

    number_of_rbf
        Number of radial basis edge features.

    target_value
        Optional scalar supervised target.

    Returns
    -------
    torch_geometric.data.Data
        Crystal graph object.
    """

    node_feature_matrix = (
        normalize_structure_node_matrix(
            structure=structure,
            normalization_statistics=
                node_normalization_statistics,
        )
    )

    (
        edge_index_array,
        edge_distance_array,
        edge_vector_array,
    ) = build_periodic_edge_list(
        structure=structure,
        cutoff_radius_A=
            cutoff_radius_A,
    )

    edge_feature_matrix = (
        build_edge_feature_matrix(
            edge_distances=
                edge_distance_array,
            cutoff_radius_A=
                cutoff_radius_A,
            number_of_rbf=
                number_of_rbf,
        )
    )

    position_matrix = np.asarray(
        [
            site.coords
            for site in structure
        ],
        dtype=np.float32,
    )

    global_descriptor_vector = (
        np.asarray(
            global_descriptor_vector,
            dtype=np.float32,
        )
        .reshape(
            1,
            -1,
        )
    )

    if target_value is None:

        target_tensor = torch.tensor(
            [float("nan")],
            dtype=torch.float32,
        )

    else:

        target_tensor = torch.tensor(
            [
                float(
                    target_value
                )
            ],
            dtype=torch.float32,
        )

    graph = Data(
        x=torch.tensor(
            node_feature_matrix,
            dtype=torch.float32,
        ),

        edge_index=torch.tensor(
            edge_index_array,
            dtype=torch.long,
        ),

        edge_attr=torch.tensor(
            edge_feature_matrix,
            dtype=torch.float32,
        ),

        pos=torch.tensor(
            position_matrix,
            dtype=torch.float32,
        ),

        graph_attr=torch.tensor(
            global_descriptor_vector,
            dtype=torch.float32,
        ),

        y=target_tensor,
    )

    graph.configuration_id = (
        configuration_id
    )

    graph.num_nodes_original = int(
        len(structure)
    )

    graph.cutoff_radius_A = float(
        cutoff_radius_A
    )

    graph.edge_vectors = torch.tensor(
        edge_vector_array,
        dtype=torch.float32,
    )

    return graph


def validate_graph(
    graph: Data,
) -> Dict[str, bool]:
    """
    Validate one PyTorch Geometric crystal graph.

    Returns
    -------
    dict
        Individual validation flags and overall validity.
    """

    edge_index_valid = bool(
        graph.edge_index.ndim == 2
        and
        graph.edge_index.shape[0] == 2
        and
        graph.edge_index.numel() > 0
        and
        graph.edge_index.min().item()
        >= 0
        and
        graph.edge_index.max().item()
        < graph.num_nodes
    )

    node_features_finite = bool(
        torch.isfinite(
            graph.x
        ).all().item()
    )

    edge_features_finite = bool(
        torch.isfinite(
            graph.edge_attr
        ).all().item()
    )

    positions_finite = bool(
        torch.isfinite(
            graph.pos
        ).all().item()
    )

    global_features_finite = bool(
        torch.isfinite(
            graph.graph_attr
        ).all().item()
    )

    has_edges = bool(
        graph.num_edges > 0
    )

    node_count_consistent = bool(
        graph.num_nodes
        == graph.num_nodes_original
    )

    edge_vector_consistent = bool(
        hasattr(
            graph,
            "edge_vectors",
        )
        and
        graph.edge_vectors.shape[0]
        == graph.num_edges
        and
        graph.edge_vectors.shape[1]
        == 3
    )

    graph_valid = all(
        [
            edge_index_valid,
            node_features_finite,
            edge_features_finite,
            positions_finite,
            global_features_finite,
            has_edges,
            node_count_consistent,
            edge_vector_consistent,
        ]
    )

    return {
        "edge_index_valid":
            edge_index_valid,

        "node_features_finite":
            node_features_finite,

        "edge_features_finite":
            edge_features_finite,

        "positions_finite":
            positions_finite,

        "global_features_finite":
            global_features_finite,

        "has_edges":
            has_edges,

        "node_count_consistent":
            node_count_consistent,

        "edge_vector_consistent":
            edge_vector_consistent,

        "graph_valid":
            bool(
                graph_valid
            ),
    }


def calculate_bidirectional_fraction(
    graph: Data,
) -> float:
    """
    Calculate the fraction of unique directed edges that have a
    reciprocal reverse connection.
    """

    source_indices = (
        graph.edge_index[0]
        .detach()
        .cpu()
        .numpy()
        .tolist()
    )

    target_indices = (
        graph.edge_index[1]
        .detach()
        .cpu()
        .numpy()
        .tolist()
    )

    edge_pairs = set(
        zip(
            source_indices,
            target_indices,
        )
    )

    if not edge_pairs:
        return 0.0

    reverse_count = sum(
        (
            target,
            source,
        )
        in edge_pairs

        for source, target
        in edge_pairs
    )

    return float(
        reverse_count
        / len(edge_pairs)
    )


def summarize_graph(
    graph: Data,
) -> dict:
    """
    Generate basic graph topology and feature summary statistics.
    """

    edge_distances = (
        graph.edge_attr[
            :,
            0
        ]
        .detach()
        .cpu()
        .numpy()
    )

    return {
        "configuration_id":
            graph.configuration_id,

        "number_of_nodes":
            int(
                graph.num_nodes
            ),

        "number_of_edges":
            int(
                graph.num_edges
            ),

        "mean_directed_degree":
            float(
                graph.num_edges
                / graph.num_nodes
            ),

        "minimum_edge_distance_A":
            float(
                edge_distances.min()
            ),

        "mean_edge_distance_A":
            float(
                edge_distances.mean()
            ),

        "maximum_edge_distance_A":
            float(
                edge_distances.max()
            ),

        "node_feature_dimension":
            int(
                graph.x.shape[1]
            ),

        "edge_feature_dimension":
            int(
                graph.edge_attr.shape[1]
            ),

        "global_feature_dimension":
            int(
                graph.graph_attr.shape[1]
            ),

        "bidirectional_fraction":
            calculate_bidirectional_fraction(
                graph
            ),
    }


def build_graph_dataset(
    structure_dataframe,
    node_normalization_statistics: Dict[str, np.ndarray],
    global_descriptor_dataframe,
    global_feature_names: Iterable[str],
    configuration_id_column: str = "configuration_id",
    structure_column: str = "structure_object",
    cutoff_radius_A: float = DEFAULT_GRAPH_CUTOFF_A,
    number_of_rbf: int = DEFAULT_NUMBER_OF_RBF,
) -> List[Data]:
    """
    Build the complete crystal graph dataset.

    Parameters
    ----------
    structure_dataframe
        Table containing structures and configuration IDs.

    global_descriptor_dataframe
        Table containing one global descriptor vector per configuration.

    global_feature_names
        Ordered global feature names.

    Returns
    -------
    list
        PyTorch Geometric Data objects.
    """

    global_feature_names = list(
        global_feature_names
    )

    graphs = []

    for _, row in (
        structure_dataframe.iterrows()
    ):

        configuration_id = row[
            configuration_id_column
        ]

        matching_global_rows = (
            global_descriptor_dataframe[
                global_descriptor_dataframe[
                    configuration_id_column
                ]
                == configuration_id
            ]
        )

        if len(
            matching_global_rows
        ) != 1:

            raise ValueError(
                "Expected exactly one global descriptor "
                f"row for {configuration_id}, found "
                f"{len(matching_global_rows)}."
            )

        global_vector = (
            matching_global_rows[
                global_feature_names
            ]
            .iloc[0]
            .to_numpy(
                dtype=np.float32,
            )
        )

        graph = structure_to_pyg_graph(
            configuration_id=
                configuration_id,

            structure=
                row[
                    structure_column
                ],

            node_normalization_statistics=
                node_normalization_statistics,

            global_descriptor_vector=
                global_vector,

            cutoff_radius_A=
                cutoff_radius_A,

            number_of_rbf=
                number_of_rbf,
        )

        validation = validate_graph(
            graph
        )

        if not validation[
            "graph_valid"
        ]:

            raise ValueError(
                f"Graph validation failed for "
                f"{configuration_id}: "
                f"{validation}"
            )

        graphs.append(
            graph
        )

    return graphs
