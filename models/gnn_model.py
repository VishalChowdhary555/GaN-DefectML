"""
Graph neural network architectures for GaN-DefectML.

Models
------
GaNDefectGNN
    Hybrid graph-level baseline using atomic node features and
    physics-informed global descriptors.

EdgeAwareGaNDefectGNN
    Final edge-aware architecture incorporating node features,
    periodic bond/edge descriptors, and graph-level physics-informed
    descriptors.

Architecture definitions are kept separate from graph construction,
training, evaluation, and explainability.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torch_geometric.nn import (
    GATv2Conv,
    global_mean_pool,
    global_max_pool,
)


# =====================================================================
# Shared helpers
# =====================================================================

def count_trainable_parameters(
    model: nn.Module,
) -> int:
    """
    Count trainable model parameters.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def _get_batch_vector(
    data,
) -> torch.Tensor:
    """
    Return PyG batch assignments.

    For an individual graph outside a DataLoader, create a zero-valued
    batch vector so pooling still works.
    """

    if hasattr(
        data,
        "batch",
    ) and data.batch is not None:

        return data.batch

    return torch.zeros(
        data.x.shape[0],
        dtype=torch.long,
        device=data.x.device,
    )


def _prepare_graph_attributes(
    data,
) -> torch.Tensor:
    """
    Return graph-level descriptors in shape [num_graphs, features].
    """

    if not hasattr(
        data,
        "graph_attr",
    ):

        raise AttributeError(
            "Graph data does not contain 'graph_attr'."
        )

    graph_attr = data.graph_attr

    if graph_attr.ndim == 1:

        graph_attr = graph_attr.unsqueeze(
            0
        )

    return graph_attr.float()


# =====================================================================
# Original hybrid baseline
# =====================================================================

class GaNDefectGNN(nn.Module):
    """
    Hybrid graph neural network baseline.

    Information channels
    --------------------
    Node features
        16-dimensional atomic descriptors.

    Graph features
        28-dimensional compact physics-informed descriptors.

    Notes
    -----
    This baseline does NOT explicitly consume edge attributes during
    message passing. It is retained for architectural comparison with
    the final edge-aware network.
    """

    def __init__(
        self,
        node_feature_dim: int = 16,
        global_feature_dim: int = 28,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        output_dim: int = 1,
    ):
        super().__init__()

        self.node_feature_dim = int(
            node_feature_dim
        )

        self.global_feature_dim = int(
            global_feature_dim
        )

        self.hidden_dim = int(
            hidden_dim
        )

        self.dropout_probability = float(
            dropout
        )

        self.output_dim = int(
            output_dim
        )

        # -------------------------------------------------------------
        # Atomic feature encoder
        # -------------------------------------------------------------

        self.node_encoder = nn.Sequential(
            nn.Linear(
                node_feature_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.LayerNorm(
                hidden_dim
            ),
        )

        # -------------------------------------------------------------
        # Graph attention message passing
        # -------------------------------------------------------------

        self.conv1 = GATv2Conv(
            in_channels=
                hidden_dim,

            out_channels=
                hidden_dim,

            heads=2,

            concat=False,
        )

        self.conv2 = GATv2Conv(
            in_channels=
                hidden_dim,

            out_channels=
                hidden_dim,

            heads=2,

            concat=False,
        )

        self.conv3 = GATv2Conv(
            in_channels=
                hidden_dim,

            out_channels=
                hidden_dim,

            heads=1,

            concat=False,
        )

        # -------------------------------------------------------------
        # Graph embedding
        #
        # Mean and max pooling each produce hidden_dim features.
        # -------------------------------------------------------------

        self.graph_projection = nn.Sequential(
            nn.Linear(
                2 * hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(
                dropout
            ),
        )

        # -------------------------------------------------------------
        # Physics-informed global descriptor encoder
        # -------------------------------------------------------------

        self.global_projection = nn.Sequential(
            nn.Linear(
                global_feature_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.LayerNorm(
                hidden_dim
            ),
            nn.Dropout(
                dropout
            ),
        )

        # -------------------------------------------------------------
        # Regression head
        # -------------------------------------------------------------

        self.output_head = nn.Sequential(
            nn.Linear(
                2 * hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                hidden_dim // 2,
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim // 2,
                output_dim,
            ),
        )


    def forward(
        self,
        data,
    ) -> torch.Tensor:
        """
        Perform a forward pass.

        Parameters
        ----------
        data
            PyTorch Geometric Data or Batch containing ``x``,
            ``edge_index``, ``graph_attr``, and optionally ``batch``.

        Returns
        -------
        torch.Tensor
            Graph-level predictions.
        """

        x = data.x.float()

        edge_index = (
            data.edge_index
        )

        batch = (
            _get_batch_vector(
                data
            )
        )

        graph_attr = (
            _prepare_graph_attributes(
                data
            )
        )

        # Node encoding
        x = self.node_encoder(
            x
        )

        # Message passing
        x = self.conv1(
            x,
            edge_index,
        )

        x = torch.relu(
            x
        )

        x = self.conv2(
            x,
            edge_index,
        )

        x = torch.relu(
            x
        )

        x = self.conv3(
            x,
            edge_index,
        )

        x = torch.relu(
            x
        )

        # Graph pooling
        mean_embedding = (
            global_mean_pool(
                x,
                batch,
            )
        )

        max_embedding = (
            global_max_pool(
                x,
                batch,
            )
        )

        graph_embedding = (
            torch.cat(
                [
                    mean_embedding,
                    max_embedding,
                ],
                dim=1,
            )
        )

        graph_embedding = (
            self.graph_projection(
                graph_embedding
            )
        )

        # Global descriptors
        global_embedding = (
            self.global_projection(
                graph_attr
            )
        )

        # Hybrid representation
        combined_embedding = (
            torch.cat(
                [
                    graph_embedding,
                    global_embedding,
                ],
                dim=1,
            )
        )

        return self.output_head(
            combined_embedding
        )


# =====================================================================
# Edge-aware message passing layer
# =====================================================================

class EdgeAwareGATBlock(nn.Module):
    """
    Edge-aware GATv2 message-passing block.

    Edge descriptors are projected into the hidden representation and
    supplied directly to GATv2Conv through ``edge_attr``.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_feature_dim: int,
        heads: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.edge_encoder = nn.Sequential(
            nn.Linear(
                edge_feature_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.LayerNorm(
                hidden_dim
            ),
        )

        self.convolution = GATv2Conv(
            in_channels=
                hidden_dim,

            out_channels=
                hidden_dim,

            heads=heads,

            concat=False,

            edge_dim=
                hidden_dim,

            dropout=
                dropout,
        )

        self.normalization = (
            nn.LayerNorm(
                hidden_dim
            )
        )

        self.activation = (
            nn.ReLU()
        )


    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply edge-aware message passing with a residual connection.
        """

        encoded_edges = (
            self.edge_encoder(
                edge_attr
            )
        )

        updated_nodes = (
            self.convolution(
                x,
                edge_index,
                edge_attr=
                    encoded_edges,
            )
        )

        # Residual connection
        updated_nodes = (
            updated_nodes
            + x
        )

        updated_nodes = (
            self.normalization(
                updated_nodes
            )
        )

        return self.activation(
            updated_nodes
        )


# =====================================================================
# Final edge-aware architecture
# =====================================================================

class EdgeAwareGaNDefectGNN(nn.Module):
    """
    Edge-aware hybrid GNN for GaN defect-property prediction.

    Input channels
    --------------
    Node features
        16 atomic descriptors.

    Edge features
        17 descriptors:
        - raw periodic interatomic distance,
        - 16 Gaussian radial basis features.

    Global features
        28 compact physics-informed defect descriptors.

    The network combines local periodic crystal information with
    global defect chemistry and structural descriptors.
    """

    def __init__(
        self,
        node_feature_dim: int = 16,
        edge_feature_dim: int = 17,
        global_feature_dim: int = 28,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        output_dim: int = 1,
    ):
        super().__init__()

        self.node_feature_dim = int(
            node_feature_dim
        )

        self.edge_feature_dim = int(
            edge_feature_dim
        )

        self.global_feature_dim = int(
            global_feature_dim
        )

        self.hidden_dim = int(
            hidden_dim
        )

        self.dropout_probability = float(
            dropout
        )

        self.output_dim = int(
            output_dim
        )

        # -------------------------------------------------------------
        # Node encoder
        # -------------------------------------------------------------

        self.node_encoder = nn.Sequential(
            nn.Linear(
                node_feature_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.LayerNorm(
                hidden_dim
            ),
        )

        # -------------------------------------------------------------
        # Edge-aware message passing
        # -------------------------------------------------------------

        self.edge_block1 = (
            EdgeAwareGATBlock(
                hidden_dim=
                    hidden_dim,

                edge_feature_dim=
                    edge_feature_dim,

                heads=2,

                dropout=
                    dropout,
            )
        )

        self.edge_block2 = (
            EdgeAwareGATBlock(
                hidden_dim=
                    hidden_dim,

                edge_feature_dim=
                    edge_feature_dim,

                heads=2,

                dropout=
                    dropout,
            )
        )

        self.edge_block3 = (
            EdgeAwareGATBlock(
                hidden_dim=
                    hidden_dim,

                edge_feature_dim=
                    edge_feature_dim,

                heads=1,

                dropout=
                    dropout,
            )
        )

        # -------------------------------------------------------------
        # Crystal graph representation
        # -------------------------------------------------------------

        self.graph_projection = nn.Sequential(
            nn.Linear(
                2 * hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(
                dropout
            ),
        )

        # -------------------------------------------------------------
        # Global physics-informed descriptor pathway
        # -------------------------------------------------------------

        self.global_projection = nn.Sequential(
            nn.Linear(
                global_feature_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.LayerNorm(
                hidden_dim
            ),
            nn.Dropout(
                dropout
            ),
        )

        # -------------------------------------------------------------
        # Prediction head
        # -------------------------------------------------------------

        self.output_head = nn.Sequential(
            nn.Linear(
                2 * hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                hidden_dim // 2,
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim // 2,
                output_dim,
            ),
        )


    def forward(
        self,
        data,
    ) -> torch.Tensor:
        """
        Predict a graph-level defect property.
        """

        if not hasattr(
            data,
            "edge_attr",
        ):

            raise AttributeError(
                "EdgeAwareGaNDefectGNN requires "
                "graph.edge_attr."
            )

        x = data.x.float()

        edge_index = (
            data.edge_index
        )

        edge_attr = (
            data.edge_attr.float()
        )

        graph_attr = (
            _prepare_graph_attributes(
                data
            )
        )

        batch = (
            _get_batch_vector(
                data
            )
        )

        # -------------------------------------------------------------
        # Atomic representation
        # -------------------------------------------------------------

        x = self.node_encoder(
            x
        )

        # -------------------------------------------------------------
        # Edge-aware periodic message passing
        # -------------------------------------------------------------

        x = self.edge_block1(
            x,
            edge_index,
            edge_attr,
        )

        x = self.edge_block2(
            x,
            edge_index,
            edge_attr,
        )

        x = self.edge_block3(
            x,
            edge_index,
            edge_attr,
        )

        # -------------------------------------------------------------
        # Graph pooling
        # -------------------------------------------------------------

        mean_embedding = (
            global_mean_pool(
                x,
                batch,
            )
        )

        max_embedding = (
            global_max_pool(
                x,
                batch,
            )
        )

        graph_embedding = (
            torch.cat(
                [
                    mean_embedding,
                    max_embedding,
                ],
                dim=1,
            )
        )

        graph_embedding = (
            self.graph_projection(
                graph_embedding
            )
        )

        # -------------------------------------------------------------
        # Physics-informed global descriptors
        # -------------------------------------------------------------

        global_embedding = (
            self.global_projection(
                graph_attr
            )
        )

        # -------------------------------------------------------------
        # Hybrid representation
        # -------------------------------------------------------------

        combined_embedding = (
            torch.cat(
                [
                    graph_embedding,
                    global_embedding,
                ],
                dim=1,
            )
        )

        prediction = (
            self.output_head(
                combined_embedding
            )
        )

        return prediction


# =====================================================================
# Model factories
# =====================================================================

def build_baseline_gnn(
    node_feature_dim: int = 16,
    global_feature_dim: int = 28,
    hidden_dim: int = 64,
    dropout: float = 0.2,
    output_dim: int = 1,
) -> GaNDefectGNN:
    """
    Build the original hybrid GNN baseline.
    """

    return GaNDefectGNN(
        node_feature_dim=
            node_feature_dim,

        global_feature_dim=
            global_feature_dim,

        hidden_dim=
            hidden_dim,

        dropout=
            dropout,

        output_dim=
            output_dim,
    )


def build_edge_aware_gnn(
    node_feature_dim: int = 16,
    edge_feature_dim: int = 17,
    global_feature_dim: int = 28,
    hidden_dim: int = 64,
    dropout: float = 0.2,
    output_dim: int = 1,
) -> EdgeAwareGaNDefectGNN:
    """
    Build the final edge-aware GaN defect GNN.
    """

    return EdgeAwareGaNDefectGNN(
        node_feature_dim=
            node_feature_dim,

        edge_feature_dim=
            edge_feature_dim,

        global_feature_dim=
            global_feature_dim,

        hidden_dim=
            hidden_dim,

        dropout=
            dropout,

        output_dim=
            output_dim,
    )


def build_default_gnn(
    edge_aware: bool = True,
):
    """
    Build the project's default GNN architecture.

    The edge-aware network is the preferred model.
    """

    if edge_aware:

        return build_edge_aware_gnn()

    return build_baseline_gnn()
