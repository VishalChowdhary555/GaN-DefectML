"""
Visualization utilities for GaN-DefectML explainability outputs.

This module provides reusable plotting helpers for:
- tabular feature importance,
- grouped descriptor importance,
- SHAP-style importance tables,
- node-level graph importance,
- edge-pair importance,
- graph-descriptor importance,
- and defect-centered importance vs distance.

Plotting functions return Matplotlib figure/axis objects so they can
be further customized or saved externally.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_feature_importance(
    importance_df: pd.DataFrame,
    feature_column: str = "feature",
    importance_column: str = "absolute_importance",
    top_n: int = 15,
    title: str = "Feature Importance",
):
    """
    Plot ranked feature importance.

    Parameters
    ----------
    importance_df
        Feature importance table.

    feature_column
        Column containing feature names.

    importance_column
        Numerical importance column.

    top_n
        Maximum number of displayed features.

    title
        Plot title.

    Returns
    -------
    tuple
        ``(figure, axis)``
    """

    if importance_df.empty:
        raise ValueError(
            "Importance table is empty."
        )

    required_columns = {
        feature_column,
        importance_column,
    }

    missing = (
        required_columns
        - set(
            importance_df.columns
        )
    )

    if missing:
        raise ValueError(
            f"Missing columns: "
            f"{sorted(missing)}"
        )

    plot_df = (
        importance_df
        .nlargest(
            top_n,
            importance_column,
        )
        .sort_values(
            importance_column,
            ascending=True,
        )
    )

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    axis.barh(
        plot_df[
            feature_column
        ],
        plot_df[
            importance_column
        ],
    )

    axis.set_xlabel(
        "Importance"
    )

    axis.set_ylabel(
        "Feature"
    )

    axis.set_title(
        title,
        weight="bold",
    )

    axis.grid(
        axis="x",
        alpha=0.25,
    )

    figure.tight_layout()

    return figure, axis


def plot_grouped_importance(
    grouped_df: pd.DataFrame,
    group_column: str = "feature_group",
    importance_column: str = "total_importance",
    title: str = "Physics-Informed Descriptor Group Importance",
):
    """
    Plot grouped physical descriptor importance.
    """

    if grouped_df.empty:
        raise ValueError(
            "Grouped importance table is empty."
        )

    plot_df = (
        grouped_df
        .sort_values(
            importance_column,
            ascending=True,
        )
    )

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    axis.barh(
        plot_df[
            group_column
        ],
        plot_df[
            importance_column
        ],
    )

    axis.set_xlabel(
        "Grouped Importance"
    )

    axis.set_ylabel(
        "Descriptor Group"
    )

    axis.set_title(
        title,
        weight="bold",
    )

    axis.grid(
        axis="x",
        alpha=0.25,
    )

    figure.tight_layout()

    return figure, axis


def plot_shap_importance(
    shap_importance_df: pd.DataFrame,
    top_n: int = 15,
):
    """
    Plot mean absolute SHAP importance.
    """

    return plot_feature_importance(
        importance_df=
            shap_importance_df,

        feature_column=
            "feature",

        importance_column=
            "mean_absolute_SHAP",

        top_n=
            top_n,

        title=
            "Mean Absolute SHAP Importance",
    )


def plot_node_importance(
    node_importance_df: pd.DataFrame,
    top_n: int = 15,
    title: str = "Graph Node Importance",
):
    """
    Plot the most influential graph nodes.
    """

    if node_importance_df.empty:
        raise ValueError(
            "Node importance table is empty."
        )

    if "importance" not in (
        node_importance_df.columns
    ):

        raise ValueError(
            "Node importance table must contain "
            "'importance'."
        )

    plot_df = (
        node_importance_df
        .nlargest(
            top_n,
            "importance",
        )
        .copy()
    )

    if "element" in plot_df.columns:

        plot_df[
            "node_label"
        ] = (
            plot_df[
                "element"
            ].astype(str)
            + " ["
            + plot_df[
                "node_index"
            ].astype(str)
            + "]"
        )

    else:

        plot_df[
            "node_label"
        ] = (
            "Node "
            + plot_df[
                "node_index"
            ].astype(str)
        )

    plot_df = (
        plot_df
        .sort_values(
            "importance",
            ascending=True,
        )
    )

    figure, axis = plt.subplots(
        figsize=(9, 7)
    )

    axis.barh(
        plot_df[
            "node_label"
        ],
        plot_df[
            "importance"
        ],
    )

    axis.set_xlabel(
        "Absolute Output Change"
    )

    axis.set_ylabel(
        "Atomic Node"
    )

    axis.set_title(
        title,
        weight="bold",
    )

    axis.grid(
        axis="x",
        alpha=0.25,
    )

    figure.tight_layout()

    return figure, axis


def plot_edge_pair_importance(
    edge_importance_df: pd.DataFrame,
    title: str = "Chemical Edge-Pair Importance",
):
    """
    Plot chemically grouped edge-pair importance.
    """

    if edge_importance_df.empty:
        raise ValueError(
            "Edge importance table is empty."
        )

    required_columns = {
        "edge_pair",
        "importance",
    }

    missing = (
        required_columns
        - set(
            edge_importance_df.columns
        )
    )

    if missing:
        raise ValueError(
            f"Missing columns: "
            f"{sorted(missing)}"
        )

    plot_df = (
        edge_importance_df
        .sort_values(
            "importance",
            ascending=True,
        )
    )

    figure, axis = plt.subplots(
        figsize=(8, 6)
    )

    axis.barh(
        plot_df[
            "edge_pair"
        ],
        plot_df[
            "importance"
        ],
    )

    axis.set_xlabel(
        "Absolute Output Change"
    )

    axis.set_ylabel(
        "Chemical Edge Pair"
    )

    axis.set_title(
        title,
        weight="bold",
    )

    axis.grid(
        axis="x",
        alpha=0.25,
    )

    figure.tight_layout()

    return figure, axis


def plot_graph_descriptor_importance(
    descriptor_importance_df: pd.DataFrame,
    top_n: int = 15,
):
    """
    Plot graph-level descriptor masking importance.
    """

    return plot_feature_importance(
        importance_df=
            descriptor_importance_df,

        feature_column=
            "feature",

        importance_column=
            "importance",

        top_n=
            top_n,

        title=
            "Graph-Level Descriptor Importance",
    )


def plot_importance_vs_defect_distance(
    node_distance_df: pd.DataFrame,
    distance_column: str = "distance_from_defect_A",
    importance_column: str = "importance",
    defect_flag_column: str = "is_explicit_defect_atom",
    title: str = "Node Importance vs Defect-Centre Distance",
):
    """
    Plot node importance as a function of periodic distance
    from a known defect center.

    This is useful for testing whether a trained model localizes
    learned importance around the defect.
    """

    required_columns = {
        distance_column,
        importance_column,
    }

    missing = (
        required_columns
        - set(
            node_distance_df.columns
        )
    )

    if missing:
        raise ValueError(
            f"Missing columns: "
            f"{sorted(missing)}"
        )

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    axis.scatter(
        node_distance_df[
            distance_column
        ],
        node_distance_df[
            importance_column
        ],
        s=70,
    )

    if (
        defect_flag_column
        in node_distance_df.columns
    ):

        defect_rows = (
            node_distance_df[
                node_distance_df[
                    defect_flag_column
                ].astype(bool)
            ]
        )

        if not defect_rows.empty:

            axis.scatter(
                defect_rows[
                    distance_column
                ],
                defect_rows[
                    importance_column
                ],
                s=220,
                marker="*",
                edgecolors="black",
                linewidth=1.0,
                label="Explicit defect atom",
            )

            axis.legend()

    axis.set_xlabel(
        "Periodic Distance from Defect Centre (Å)"
    )

    axis.set_ylabel(
        "Node Importance"
    )

    axis.set_title(
        title,
        weight="bold",
    )

    axis.grid(
        alpha=0.25,
    )

    figure.tight_layout()

    return figure, axis


def plot_prediction_scatter(
    prediction_df: pd.DataFrame,
    true_column: str = "true_value",
    predicted_column: str = "predicted_value",
    title: str = "Predicted vs Actual",
):
    """
    Plot predicted values against validated target values.
    """

    required_columns = {
        true_column,
        predicted_column,
    }

    missing = (
        required_columns
        - set(
            prediction_df.columns
        )
    )

    if missing:
        raise ValueError(
            f"Missing columns: "
            f"{sorted(missing)}"
        )

    true_values = (
        prediction_df[
            true_column
        ]
        .to_numpy(
            dtype=float
        )
    )

    predicted_values = (
        prediction_df[
            predicted_column
        ]
        .to_numpy(
            dtype=float
        )
    )

    lower_bound = float(
        min(
            np.min(
                true_values
            ),
            np.min(
                predicted_values
            ),
        )
    )

    upper_bound = float(
        max(
            np.max(
                true_values
            ),
            np.max(
                predicted_values
            ),
        )
    )

    figure, axis = plt.subplots(
        figsize=(7, 7)
    )

    axis.scatter(
        true_values,
        predicted_values,
        s=70,
    )

    axis.plot(
        [
            lower_bound,
            upper_bound,
        ],
        [
            lower_bound,
            upper_bound,
        ],
        linestyle="--",
    )

    axis.set_xlabel(
        "Actual"
    )

    axis.set_ylabel(
        "Predicted"
    )

    axis.set_title(
        title,
        weight="bold",
    )

    axis.grid(
        alpha=0.25,
    )

    figure.tight_layout()

    return figure, axis


def save_figure(
    figure,
    output_path,
    dpi: int = 300,
):
    """
    Save a Matplotlib figure using publication-friendly settings.
    """

    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    return output_path
