"""
Training utilities for graph neural networks in GaN-DefectML.

Responsibilities
----------------
- identify graphs containing validated targets,
- enforce supervised-learning readiness requirements,
- construct graph data loaders,
- train regression GNNs,
- perform early stopping,
- evaluate graph-level regression models,
- generate predictions,
- and support cross-validation.

The neural-network architecture itself is intentionally kept outside
this module in the top-level ``models`` package.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from sklearn.model_selection import KFold

from torch_geometric.loader import DataLoader


DEFAULT_RANDOM_SEED = 42


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_random_seed(
    seed: int = DEFAULT_RANDOM_SEED,
) -> None:
    """
    Set random seeds used by NumPy and PyTorch.
    """

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Target handling
# ---------------------------------------------------------------------

def graph_has_valid_target(
    graph,
) -> bool:
    """
    Check whether a graph contains a finite scalar target.
    """

    if not hasattr(graph, "y"):
        return False

    if graph.y is None:
        return False

    if graph.y.numel() == 0:
        return False

    return bool(
        torch.isfinite(
            graph.y
        ).all().item()
    )


def get_labeled_graphs(
    graphs,
):
    """
    Return only graphs containing finite supervised targets.
    """

    return [
        graph
        for graph in graphs
        if graph_has_valid_target(
            graph
        )
    ]


def count_labeled_graphs(
    graphs,
) -> int:
    """
    Count graphs containing valid supervised labels.
    """

    return len(
        get_labeled_graphs(
            graphs
        )
    )


def assess_gnn_regression_readiness(
    graphs,
    minimum_samples: int = 30,
    minimum_unique_values: int = 5,
) -> dict:
    """
    Determine whether graph regression is scientifically ready.

    This prevents model fitting when only a tiny number of defect
    calculations or placeholder labels are available.
    """

    labeled_graphs = (
        get_labeled_graphs(
            graphs
        )
    )

    target_values = np.asarray(
        [
            float(
                graph.y
                .detach()
                .cpu()
                .reshape(-1)[0]
                .item()
            )
            for graph in labeled_graphs
        ],
        dtype=float,
    )

    number_of_samples = len(
        target_values
    )

    number_of_unique_values = (
        len(
            np.unique(
                target_values
            )
        )
        if number_of_samples > 0
        else 0
    )

    is_ready = bool(
        number_of_samples
        >= minimum_samples
        and
        number_of_unique_values
        >= minimum_unique_values
    )

    return {
        "is_ready":
            is_ready,

        "available_samples":
            int(
                number_of_samples
            ),

        "unique_values":
            int(
                number_of_unique_values
            ),

        "minimum_samples":
            int(
                minimum_samples
            ),

        "minimum_unique_values":
            int(
                minimum_unique_values
            ),
    }


# ---------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------

def create_graph_loader(
    graphs,
    batch_size: int = 8,
    shuffle: bool = False,
) -> DataLoader:
    """
    Construct a PyTorch Geometric DataLoader.
    """

    if len(graphs) == 0:
        raise ValueError(
            "Cannot create a graph loader "
            "from an empty graph list."
        )

    return DataLoader(
        graphs,
        batch_size=batch_size,
        shuffle=shuffle,
    )


# ---------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------

def move_batch_to_device(
    batch,
    device: torch.device,
):
    """
    Move a PyG graph batch to the selected device.
    """

    return batch.to(
        device
    )


def extract_batch_targets(
    batch,
) -> torch.Tensor:
    """
    Extract graph-level scalar regression targets.
    """

    targets = batch.y.float()

    targets = targets.reshape(
        -1
    )

    return targets


# ---------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------

def calculate_gnn_regression_metrics(
    y_true,
    y_pred,
) -> dict:
    """
    Calculate MAE, RMSE, and R2 for graph regression.
    """

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = (
        r2_score(
            y_true,
            y_pred,
        )
        if len(y_true) >= 2
        else np.nan
    )

    return {
        "MAE":
            float(mae),

        "RMSE":
            float(rmse),

        "R2":
            float(r2),
    }


# ---------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------

def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer,
    loss_function,
    device: torch.device,
) -> float:
    """
    Train a GNN for one epoch.

    Returns
    -------
    float
        Mean sample-weighted training loss.
    """

    model.train()

    total_loss = 0.0
    total_graphs = 0

    for batch in loader:

        batch = move_batch_to_device(
            batch,
            device,
        )

        targets = (
            extract_batch_targets(
                batch
            )
        )

        if not torch.isfinite(
            targets
        ).all():

            raise ValueError(
                "Non-finite targets encountered "
                "during GNN training."
            )

        optimizer.zero_grad()

        predictions = model(
            batch
        ).reshape(
            -1
        )

        if predictions.shape != (
            targets.shape
        ):

            raise ValueError(
                "Prediction and target shapes "
                f"do not match: "
                f"{tuple(predictions.shape)} vs "
                f"{tuple(targets.shape)}."
            )

        loss = loss_function(
            predictions,
            targets,
        )

        if not torch.isfinite(
            loss
        ):

            raise FloatingPointError(
                "Non-finite loss encountered "
                "during GNN training."
            )

        loss.backward()

        optimizer.step()

        number_of_graphs = int(
            batch.num_graphs
        )

        total_loss += (
            float(
                loss.item()
            )
            * number_of_graphs
        )

        total_graphs += (
            number_of_graphs
        )

    if total_graphs == 0:
        raise RuntimeError(
            "No graphs were processed "
            "during training."
        )

    return (
        total_loss
        / total_graphs
    )


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

@torch.no_grad()
def evaluate_gnn(
    model,
    loader: DataLoader,
    loss_function,
    device: torch.device,
) -> dict:
    """
    Evaluate a graph-regression model.
    """

    model.eval()

    total_loss = 0.0
    total_graphs = 0

    all_targets = []
    all_predictions = []

    for batch in loader:

        batch = move_batch_to_device(
            batch,
            device,
        )

        targets = (
            extract_batch_targets(
                batch
            )
        )

        predictions = model(
            batch
        ).reshape(
            -1
        )

        loss = loss_function(
            predictions,
            targets,
        )

        number_of_graphs = int(
            batch.num_graphs
        )

        total_loss += (
            float(
                loss.item()
            )
            * number_of_graphs
        )

        total_graphs += (
            number_of_graphs
        )

        all_targets.extend(
            targets
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

        all_predictions.extend(
            predictions
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

    if total_graphs == 0:
        raise RuntimeError(
            "No graphs were processed "
            "during evaluation."
        )

    metrics = (
        calculate_gnn_regression_metrics(
            y_true=
                all_targets,

            y_pred=
                all_predictions,
        )
    )

    return {
        "loss":
            float(
                total_loss
                / total_graphs
            ),

        "metrics":
            metrics,

        "targets":
            np.asarray(
                all_targets,
                dtype=float,
            ),

        "predictions":
            np.asarray(
                all_predictions,
                dtype=float,
            ),
    }


# ---------------------------------------------------------------------
# Model fitting with early stopping
# ---------------------------------------------------------------------

def fit_gnn_regressor(
    model,
    train_graphs,
    validation_graphs,
    device: torch.device,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    maximum_epochs: int = 300,
    patience: int = 30,
    minimum_delta: float = 1e-6,
    verbose: bool = True,
) -> dict:
    """
    Fit a graph-regression model using validation-loss early stopping.
    """

    if len(train_graphs) == 0:
        raise ValueError(
            "Training graph set is empty."
        )

    if len(validation_graphs) == 0:
        raise ValueError(
            "Validation graph set is empty."
        )

    train_loader = (
        create_graph_loader(
            train_graphs,
            batch_size=batch_size,
            shuffle=True,
        )
    )

    validation_loader = (
        create_graph_loader(
            validation_graphs,
            batch_size=batch_size,
            shuffle=False,
        )
    )

    model = model.to(
        device
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    loss_function = (
        torch.nn.MSELoss()
    )

    best_validation_loss = (
        np.inf
    )

    best_model_state = None

    epochs_without_improvement = 0

    history_records = []

    for epoch in range(
        1,
        maximum_epochs + 1,
    ):

        training_loss = (
            train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                loss_function=
                    loss_function,
                device=device,
            )
        )

        validation_result = (
            evaluate_gnn(
                model=model,
                loader=
                    validation_loader,
                loss_function=
                    loss_function,
                device=device,
            )
        )

        validation_loss = (
            validation_result[
                "loss"
            ]
        )

        history_records.append(
            {
                "epoch":
                    epoch,

                "training_loss":
                    training_loss,

                "validation_loss":
                    validation_loss,

                "validation_MAE":
                    validation_result[
                        "metrics"
                    ][
                        "MAE"
                    ],

                "validation_RMSE":
                    validation_result[
                        "metrics"
                    ][
                        "RMSE"
                    ],

                "validation_R2":
                    validation_result[
                        "metrics"
                    ][
                        "R2"
                    ],
            }
        )

        improved = (
            validation_loss
            <
            (
                best_validation_loss
                - minimum_delta
            )
        )

        if improved:

            best_validation_loss = (
                validation_loss
            )

            best_model_state = (
                deepcopy(
                    model.state_dict()
                )
            )

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if verbose and (
            epoch == 1
            or epoch % 25 == 0
        ):

            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: "
                f"{training_loss:.6f} | "
                f"Val Loss: "
                f"{validation_loss:.6f} | "
                f"Val MAE: "
                f"{validation_result['metrics']['MAE']:.6f}"
            )

        if (
            epochs_without_improvement
            >= patience
        ):

            if verbose:

                print(
                    "Early stopping triggered "
                    f"at epoch {epoch}."
                )

            break

    if best_model_state is None:
        raise RuntimeError(
            "Training completed without a "
            "valid model checkpoint."
        )

    model.load_state_dict(
        best_model_state
    )

    return {
        "model":
            model,

        "history":
            pd.DataFrame(
                history_records
            ),

        "best_validation_loss":
            float(
                best_validation_loss
            ),

        "epochs_trained":
            int(
                len(
                    history_records
                )
            ),
    }


# ---------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------

def cross_validate_gnn_regressor(
    graphs,
    model_factory: Callable,
    device: torch.device,
    n_splits: int = 5,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    maximum_epochs: int = 300,
    patience: int = 30,
    random_state: int = DEFAULT_RANDOM_SEED,
) -> dict:
    """
    Perform K-fold cross-validation for graph regression.

    ``model_factory`` must return a fresh, untrained model each time
    it is called.
    """

    labeled_graphs = (
        get_labeled_graphs(
            graphs
        )
    )

    if len(labeled_graphs) < (
        n_splits
    ):

        raise ValueError(
            "Number of labeled graphs must "
            "be at least the number of CV folds."
        )

    set_random_seed(
        random_state
    )

    splitter = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    graph_indices = np.arange(
        len(
            labeled_graphs
        )
    )

    fold_records = []

    prediction_records = []

    for fold_number, (
        train_indices,
        validation_indices,
    ) in enumerate(
        splitter.split(
            graph_indices
        ),
        start=1,
    ):

        train_graphs = [
            labeled_graphs[
                index
            ]
            for index
            in train_indices
        ]

        validation_graphs = [
            labeled_graphs[
                index
            ]
            for index
            in validation_indices
        ]

        model = model_factory()

        fit_result = (
            fit_gnn_regressor(
                model=model,
                train_graphs=
                    train_graphs,
                validation_graphs=
                    validation_graphs,
                device=device,
                batch_size=
                    batch_size,
                learning_rate=
                    learning_rate,
                weight_decay=
                    weight_decay,
                maximum_epochs=
                    maximum_epochs,
                patience=
                    patience,
                verbose=False,
            )
        )

        validation_loader = (
            create_graph_loader(
                validation_graphs,
                batch_size=
                    batch_size,
                shuffle=False,
            )
        )

        evaluation = (
            evaluate_gnn(
                model=
                    fit_result[
                        "model"
                    ],
                loader=
                    validation_loader,
                loss_function=
                    torch.nn.MSELoss(),
                device=device,
            )
        )

        fold_records.append(
            {
                "fold":
                    fold_number,

                "MAE":
                    evaluation[
                        "metrics"
                    ][
                        "MAE"
                    ],

                "RMSE":
                    evaluation[
                        "metrics"
                    ][
                        "RMSE"
                    ],

                "R2":
                    evaluation[
                        "metrics"
                    ][
                        "R2"
                    ],

                "epochs_trained":
                    fit_result[
                        "epochs_trained"
                    ],
            }
        )

        for graph, true_value, predicted_value in zip(
            validation_graphs,
            evaluation[
                "targets"
            ],
            evaluation[
                "predictions"
            ],
        ):

            prediction_records.append(
                {
                    "fold":
                        fold_number,

                    "configuration_id":
                        getattr(
                            graph,
                            "configuration_id",
                            None,
                        ),

                    "true_value":
                        float(
                            true_value
                        ),

                    "predicted_value":
                        float(
                            predicted_value
                        ),
                }
            )

    prediction_df = pd.DataFrame(
        prediction_records
    )

    overall_metrics = (
        calculate_gnn_regression_metrics(
            y_true=
                prediction_df[
                    "true_value"
                ],

            y_pred=
                prediction_df[
                    "predicted_value"
                ],
        )
    )

    return {
        "fold_metrics":
            pd.DataFrame(
                fold_records
            ),

        "predictions":
            prediction_df,

        "overall_metrics":
            overall_metrics,
    }


# ---------------------------------------------------------------------
# High-level scientific gate
# ---------------------------------------------------------------------

def run_gnn_regression_pipeline(
    graphs,
    model_factory: Callable,
    device: torch.device,
    minimum_samples: int = 30,
    minimum_unique_values: int = 5,
    n_splits: int = 5,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    maximum_epochs: int = 300,
    patience: int = 30,
    random_state: int = DEFAULT_RANDOM_SEED,
) -> Optional[dict]:
    """
    Run the scientifically gated GNN regression pipeline.

    Training is skipped when validated graph labels are insufficient.
    """

    readiness = (
        assess_gnn_regression_readiness(
            graphs=graphs,
            minimum_samples=
                minimum_samples,
            minimum_unique_values=
                minimum_unique_values,
        )
    )

    if not readiness[
        "is_ready"
    ]:

        print(
            "GNN training skipped: "
            "validated target coverage "
            "is insufficient."
        )

        print(
            "Available labels: "
            f"{readiness['available_samples']}"
        )

        print(
            "Minimum required: "
            f"{readiness['minimum_samples']}"
        )

        return None

    if (
        readiness[
            "available_samples"
        ]
        < n_splits
    ):

        raise ValueError(
            "Insufficient labeled graphs "
            "for the requested number of "
            "cross-validation folds."
        )

    return (
        cross_validate_gnn_regressor(
            graphs=graphs,
            model_factory=
                model_factory,
            device=device,
            n_splits=n_splits,
            batch_size=
                batch_size,
            learning_rate=
                learning_rate,
            weight_decay=
                weight_decay,
            maximum_epochs=
                maximum_epochs,
            patience=
                patience,
            random_state=
                random_state,
        )
    )
