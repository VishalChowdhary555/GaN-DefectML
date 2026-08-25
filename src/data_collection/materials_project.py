"""
Materials Project data retrieval utilities for GaN-DefectML.

This module provides helper functions for querying Ga-N materials from the
Materials Project database and converting returned documents into a structured
pandas DataFrame.
"""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

from mp_api.client import MPRester


DEFAULT_FIELDS = [
    "material_id",
    "formula_pretty",
    "chemsys",
    "composition",
    "symmetry",
    "band_gap",
    "is_metal",
    "is_stable",
    "formation_energy_per_atom",
    "energy_above_hull",
    "density",
    "volume",
    "nsites",
    "structure",
]


def fetch_gan_entries(
    api_key: str,
    chemical_system: str = "Ga-N",
    fields: Optional[Iterable[str]] = None,
):
    """
    Retrieve Ga-N materials from the Materials Project.

    Parameters
    ----------
    api_key : str
        Materials Project API key.

    chemical_system : str, default="Ga-N"
        Chemical system used for the Materials Project query.

    fields : iterable of str, optional
        Fields to request from the Materials Project API.
        If None, DEFAULT_FIELDS is used.

    Returns
    -------
    list
        Materials Project summary documents.
    """

    requested_fields = list(fields or DEFAULT_FIELDS)

    with MPRester(api_key) as mpr:
        documents = mpr.materials.summary.search(
            chemsys=[chemical_system],
            fields=requested_fields,
        )

    return documents


def _extract_symmetry_information(symmetry):
    """
    Extract crystal-system and space-group information safely.

    Parameters
    ----------
    symmetry
        Materials Project symmetry object.

    Returns
    -------
    tuple
        crystal_system, space_group_symbol, space_group_number
    """

    if symmetry is None:
        return None, None, None

    crystal_system = getattr(
        symmetry,
        "crystal_system",
        None,
    )

    space_group_symbol = getattr(
        symmetry,
        "symbol",
        None,
    )

    space_group_number = getattr(
        symmetry,
        "number",
        None,
    )

    if crystal_system is not None:
        crystal_system = str(crystal_system)

    return (
        crystal_system,
        space_group_symbol,
        space_group_number,
    )


def documents_to_dataframe(documents) -> pd.DataFrame:
    """
    Convert Materials Project summary documents into a clean DataFrame.

    Parameters
    ----------
    documents : iterable
        Materials Project summary documents.

    Returns
    -------
    pandas.DataFrame
        Standardized Ga-N materials dataset.
    """

    records = []

    for document in documents:
        (
            crystal_system,
            space_group_symbol,
            space_group_number,
        ) = _extract_symmetry_information(
            getattr(document, "symmetry", None)
        )

        material_id = getattr(
            document,
            "material_id",
            None,
        )

        if material_id is not None:
            material_id = str(material_id)

        composition = getattr(
            document,
            "composition",
            None,
        )

        record = {
            "material_id": material_id,
            "formula": getattr(
                document,
                "formula_pretty",
                None,
            ),
            "chemical_system": getattr(
                document,
                "chemsys",
                None,
            ),
            "composition": (
                str(composition)
                if composition is not None
                else None
            ),
            "crystal_system": crystal_system,
            "space_group_symbol": (
                space_group_symbol
            ),
            "space_group_number": (
                space_group_number
            ),
            "band_gap_eV": getattr(
                document,
                "band_gap",
                None,
            ),
            "is_metal": getattr(
                document,
                "is_metal",
                None,
            ),
            "is_stable": getattr(
                document,
                "is_stable",
                None,
            ),
            "formation_energy_eV_atom": getattr(
                document,
                "formation_energy_per_atom",
                None,
            ),
            "energy_above_hull_eV_atom": getattr(
                document,
                "energy_above_hull",
                None,
            ),
            "density_g_cm3": getattr(
                document,
                "density",
                None,
            ),
            "volume_A3": getattr(
                document,
                "volume",
                None,
            ),
            "number_of_sites": getattr(
                document,
                "nsites",
                None,
            ),
            "structure": getattr(
                document,
                "structure",
                None,
            ),
        }

        records.append(record)

    dataframe = pd.DataFrame(records)

    if dataframe.empty:
        return dataframe

    dataframe = (
        dataframe
        .drop_duplicates(
            subset=["material_id"]
        )
        .reset_index(drop=True)
    )

    return dataframe


def load_gan_dataset(
    api_key: str,
    chemical_system: str = "Ga-N",
    fields: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Retrieve and format the complete Ga-N dataset.

    This is the main convenience function intended for external use.

    Parameters
    ----------
    api_key : str
        Materials Project API key.

    chemical_system : str, default="Ga-N"
        Chemical system to query.

    fields : iterable of str, optional
        Optional custom Materials Project fields.

    Returns
    -------
    pandas.DataFrame
        Standardized Ga-N materials dataset.
    """

    documents = fetch_gan_entries(
        api_key=api_key,
        chemical_system=chemical_system,
        fields=fields,
    )

    dataframe = documents_to_dataframe(
        documents
    )

    return dataframe
