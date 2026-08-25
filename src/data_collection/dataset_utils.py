"""
Dataset utilities for GaN-DefectML.

Provides reusable helpers for:
- dataset validation,
- physical filtering,
- structure classification,
- numerical summaries,
- representative structure selection,
- and CSV export.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


REQUIRED_GAN_COLUMNS = {
    "material_id",
    "formula",
    "chemical_system",
    "crystal_system",
    "space_group_symbol",
    "space_group_number",
    "band_gap_eV",
    "is_metal",
    "is_stable",
    "formation_energy_eV_atom",
    "energy_above_hull_eV_atom",
    "density_g_cm3",
    "volume_A3",
    "number_of_sites",
    "structure",
}


def validate_gan_dataframe(
    dataframe: pd.DataFrame,
    required_columns: Optional[Iterable[str]] = None,
) -> None:
    """
    Validate that the dataset contains the expected GaN columns.

    Raises
    ------
    ValueError
        If required columns are missing.
    """

    required = set(
        required_columns
        if required_columns is not None
        else REQUIRED_GAN_COLUMNS
    )

    missing_columns = required - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "GaN dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )


def clean_gan_dataframe(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply lightweight cleaning to the GaN materials dataset.

    The structure column is preserved as a Pymatgen Structure object.
    """

    validate_gan_dataframe(dataframe)

    cleaned = dataframe.copy()

    cleaned = cleaned.drop_duplicates(
        subset=["material_id"]
    )

    numerical_columns = [
        "space_group_number",
        "band_gap_eV",
        "formation_energy_eV_atom",
        "energy_above_hull_eV_atom",
        "density_g_cm3",
        "volume_A3",
        "number_of_sites",
    ]

    for column in numerical_columns:
        cleaned[column] = pd.to_numeric(
            cleaned[column],
            errors="coerce",
        )

    cleaned = cleaned.reset_index(drop=True)

    return cleaned


def summarize_gan_dataset(
    dataframe: pd.DataFrame,
) -> dict:
    """
    Produce a compact summary of the GaN dataset.

    Returns
    -------
    dict
        Dataset size, crystal-system counts, stability counts,
        metallicity counts, and numerical statistics.
    """

    validate_gan_dataframe(dataframe)

    numerical_columns = [
        "band_gap_eV",
        "formation_energy_eV_atom",
        "energy_above_hull_eV_atom",
        "density_g_cm3",
        "volume_A3",
        "number_of_sites",
    ]

    summary = {
        "number_of_entries": len(dataframe),
        "number_of_columns": dataframe.shape[1],
        "formula_counts": (
            dataframe["formula"]
            .value_counts(dropna=False)
            .to_dict()
        ),
        "crystal_system_counts": (
            dataframe["crystal_system"]
            .value_counts(dropna=False)
            .to_dict()
        ),
        "space_group_counts": (
            dataframe[
                [
                    "space_group_symbol",
                    "space_group_number",
                ]
            ]
            .value_counts(dropna=False)
            .to_dict()
        ),
        "stability_counts": (
            dataframe["is_stable"]
            .value_counts(dropna=False)
            .to_dict()
        ),
        "metallicity_counts": (
            dataframe["is_metal"]
            .value_counts(dropna=False)
            .to_dict()
        ),
        "numerical_statistics": (
            dataframe[numerical_columns]
            .describe()
            .to_dict()
        ),
    }

    return summary


def classify_physical_candidate(
    row: pd.Series,
    low_energy_threshold: float = 0.05,
    ordered_metastable_threshold: float = 0.50,
) -> str:
    """
    Assign a physically motivated class to a GaN structure.

    Classification is based primarily on energy above hull and stability.

    Parameters
    ----------
    row : pandas.Series
        Row containing at least ``is_stable`` and
        ``energy_above_hull_eV_atom``.

    low_energy_threshold : float
        Maximum energy above hull for a low-energy metastable candidate.

    ordered_metastable_threshold : float
        Maximum energy above hull for an ordered metastable polymorph.

    Returns
    -------
    str
        Physical classification label.
    """

    is_stable = bool(row.get("is_stable", False))

    hull_energy = row.get(
        "energy_above_hull_eV_atom",
        np.nan,
    )

    if pd.isna(hull_energy):
        return "Unclassified"

    if is_stable or np.isclose(
        hull_energy,
        0.0,
        atol=1e-8,
    ):
        return "Ground-state candidate"

    if hull_energy <= low_energy_threshold:
        return "Low-energy metastable candidate"

    if hull_energy <= ordered_metastable_threshold:
        return "Ordered metastable polymorph"

    return "High-energy structure"


def add_physical_classification(
    dataframe: pd.DataFrame,
    low_energy_threshold: float = 0.05,
    ordered_metastable_threshold: float = 0.50,
) -> pd.DataFrame:
    """
    Add a physical-class label to every structure.
    """

    classified = dataframe.copy()

    classified["physical_class"] = classified.apply(
        lambda row: classify_physical_candidate(
            row,
            low_energy_threshold=low_energy_threshold,
            ordered_metastable_threshold=
                ordered_metastable_threshold,
        ),
        axis=1,
    )

    return classified


def filter_ordered_structures(
    dataframe: pd.DataFrame,
    maximum_sites: int = 20,
    maximum_hull_energy: Optional[float] = 0.50,
) -> pd.DataFrame:
    """
    Filter out very large or highly metastable structures.

    This is useful for isolating ordered GaN polymorphs from large
    low-symmetry supercell-like entries.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input GaN dataset.

    maximum_sites : int, default=20
        Maximum number of sites allowed.

    maximum_hull_energy : float or None, default=0.50
        Maximum energy above hull. If None, no hull-energy filter is used.

    Returns
    -------
    pandas.DataFrame
        Filtered ordered-structure subset.
    """

    filtered = dataframe[
        dataframe["number_of_sites"] <= maximum_sites
    ].copy()

    if maximum_hull_energy is not None:
        filtered = filtered[
            filtered["energy_above_hull_eV_atom"]
            <= maximum_hull_energy
        ]

    filtered = (
        filtered
        .sort_values(
            "energy_above_hull_eV_atom",
            ascending=True,
        )
        .reset_index(drop=True)
    )

    return filtered


def select_representative_structures(
    dataframe: pd.DataFrame,
    grouping_columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Select one representative structure per crystallographic group.

    By default, structures are grouped by crystal system, space group,
    and number of sites. The lowest-energy structure in each group is kept.
    """

    if grouping_columns is None:
        grouping_columns = [
            "crystal_system",
            "space_group_symbol",
            "space_group_number",
            "number_of_sites",
        ]

    grouping_columns = list(grouping_columns)

    sorted_df = dataframe.sort_values(
        "energy_above_hull_eV_atom",
        ascending=True,
    )

    representatives = (
        sorted_df
        .drop_duplicates(
            subset=grouping_columns,
            keep="first",
        )
        .reset_index(drop=True)
    )

    return representatives


def build_host_recommendation_table(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign study priorities to ordered GaN polymorphs.

    The mapping follows the project workflow:
    - mp-804: primary defect host
    - mp-830: comparison polymorph
    - mp-1181864: secondary structural reference
    - mp-1007824: exploratory reference
    - mp-2646978: exploratory reference
    - mp-2853: high-pressure structural reference

    Unknown entries are marked as unclassified.
    """

    recommendation_map = {
        "mp-804": "Primary defect host",
        "mp-830": "Comparison polymorph",
        "mp-1181864": "Secondary structural reference",
        "mp-1007824": "Exploratory reference",
        "mp-2646978": "Exploratory reference",
        "mp-2853": "High-pressure structural reference",
    }

    ranked = dataframe.copy()

    ranked["host_recommendation"] = (
        ranked["material_id"]
        .map(recommendation_map)
        .fillna("Unclassified")
    )

    ranked = ranked.sort_values(
        "energy_above_hull_eV_atom",
        ascending=True,
    ).reset_index(drop=True)

    return ranked


def export_dataframe(
    dataframe: pd.DataFrame,
    output_path: str | Path,
    index: bool = False,
) -> Path:
    """
    Export a DataFrame to CSV, creating parent directories if needed.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output_path,
        index=index,
    )

    return output_path
