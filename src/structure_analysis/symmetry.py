"""
Symmetry analysis utilities for GaN-DefectML.

Provides helpers for:
- space-group determination,
- symmetry consistency checks,
- crystal-system identification,
- symmetry-aware grouping,
- and representative structure selection.
"""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def analyze_structure_symmetry(
    structure,
    symprec: float = 0.01,
    angle_tolerance: float = 5.0,
) -> dict:
    """
    Analyze the crystallographic symmetry of a Pymatgen Structure.

    Parameters
    ----------
    structure
        Pymatgen Structure object.

    symprec : float, default=0.01
        Cartesian tolerance used for symmetry detection.

    angle_tolerance : float, default=5.0
        Angular tolerance in degrees.

    Returns
    -------
    dict
        Calculated symmetry information.
    """

    analyzer = SpacegroupAnalyzer(
        structure,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )

    return {
        "calculated_space_group":
            analyzer.get_space_group_symbol(),

        "calculated_space_group_number":
            analyzer.get_space_group_number(),

        "calculated_crystal_system":
            analyzer.get_crystal_system(),

        "is_centrosymmetric":
            analyzer.is_laue(),
    }


def validate_symmetry_against_database(
    dataframe: pd.DataFrame,
    structure_column: str = "structure",
    space_group_symbol_column: str = "space_group_symbol",
    space_group_number_column: str = "space_group_number",
    material_id_column: str = "material_id",
    symprec: float = 0.01,
    angle_tolerance: float = 5.0,
) -> pd.DataFrame:
    """
    Recalculate symmetry and compare it with database metadata.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input materials dataset.

    structure_column : str
        Column containing Pymatgen Structure objects.

    space_group_symbol_column : str
        Database space-group symbol column.

    space_group_number_column : str
        Database space-group number column.

    material_id_column : str
        Material identifier column.

    Returns
    -------
    pandas.DataFrame
        Symmetry validation table.
    """

    validation_records = []

    for _, row in dataframe.iterrows():

        structure = row[structure_column]

        calculated = analyze_structure_symmetry(
            structure=structure,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
        )

        database_symbol = row[
            space_group_symbol_column
        ]

        database_number = row[
            space_group_number_column
        ]

        calculated_symbol = calculated[
            "calculated_space_group"
        ]

        calculated_number = calculated[
            "calculated_space_group_number"
        ]

        symmetry_consistent = bool(
            str(database_symbol)
            == str(calculated_symbol)
            and int(database_number)
            == int(calculated_number)
        )

        validation_records.append(
            {
                "material_id":
                    row[material_id_column],

                "database_space_group":
                    database_symbol,

                "calculated_space_group":
                    calculated_symbol,

                "database_space_group_number":
                    database_number,

                "calculated_space_group_number":
                    calculated_number,

                "calculated_crystal_system":
                    calculated[
                        "calculated_crystal_system"
                    ],

                "is_symmetry_consistent":
                    symmetry_consistent,
            }
        )

    return pd.DataFrame(
        validation_records
    )


def build_symmetry_signature(
    structure,
    symprec: float = 0.01,
    angle_tolerance: float = 5.0,
) -> tuple:
    """
    Build a compact crystallographic symmetry signature.

    The signature can be used for grouping structures with
    comparable symmetry metadata.

    Returns
    -------
    tuple
        (
            crystal_system,
            space_group_symbol,
            space_group_number,
            number_of_sites
        )
    """

    symmetry = analyze_structure_symmetry(
        structure=structure,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )

    return (
        symmetry["calculated_crystal_system"],
        symmetry["calculated_space_group"],
        symmetry["calculated_space_group_number"],
        len(structure),
    )


def add_symmetry_signatures(
    dataframe: pd.DataFrame,
    structure_column: str = "structure",
    symprec: float = 0.01,
    angle_tolerance: float = 5.0,
) -> pd.DataFrame:
    """
    Add calculated symmetry metadata and a symmetry signature.
    """

    output = dataframe.copy()

    calculated_crystal_systems = []
    calculated_symbols = []
    calculated_numbers = []
    signatures = []

    for structure in output[
        structure_column
    ]:

        symmetry = analyze_structure_symmetry(
            structure=structure,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
        )

        calculated_crystal_systems.append(
            symmetry[
                "calculated_crystal_system"
            ]
        )

        calculated_symbols.append(
            symmetry[
                "calculated_space_group"
            ]
        )

        calculated_numbers.append(
            symmetry[
                "calculated_space_group_number"
            ]
        )

        signatures.append(
            (
                symmetry[
                    "calculated_crystal_system"
                ],
                symmetry[
                    "calculated_space_group"
                ],
                symmetry[
                    "calculated_space_group_number"
                ],
                len(structure),
            )
        )

    output[
        "calculated_crystal_system"
    ] = calculated_crystal_systems

    output[
        "calculated_space_group"
    ] = calculated_symbols

    output[
        "calculated_space_group_number"
    ] = calculated_numbers

    output[
        "symmetry_signature"
    ] = signatures

    return output


def symmetry_aware_grouping(
    dataframe: pd.DataFrame,
    structure_column: str = "structure",
    symprec: float = 0.01,
    angle_tolerance: float = 5.0,
) -> pd.DataFrame:
    """
    Assign a symmetry-aware structure-group identifier.

    Structures with the same calculated crystal system, space group,
    and number of sites are assigned the same group.

    This is intentionally conservative and does not claim strict
    structural equivalence.
    """

    grouped = add_symmetry_signatures(
        dataframe=dataframe,
        structure_column=structure_column,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )

    signature_to_group = {}

    next_group_id = 1
    group_ids = []

    for signature in grouped[
        "symmetry_signature"
    ]:

        if signature not in signature_to_group:

            signature_to_group[
                signature
            ] = next_group_id

            next_group_id += 1

        group_ids.append(
            signature_to_group[
                signature
            ]
        )

    grouped[
        "symmetry_group"
    ] = group_ids

    return grouped


def select_symmetry_representatives(
    dataframe: pd.DataFrame,
    energy_column: str = "energy_above_hull_eV_atom",
) -> pd.DataFrame:
    """
    Select one lowest-energy representative per symmetry group.

    The input DataFrame must already contain a ``symmetry_group`` column.
    """

    if "symmetry_group" not in dataframe.columns:
        raise ValueError(
            "DataFrame must contain 'symmetry_group'. "
            "Run symmetry_aware_grouping() first."
        )

    representatives = (
        dataframe
        .sort_values(
            energy_column,
            ascending=True,
        )
        .drop_duplicates(
            subset=["symmetry_group"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    return representatives


def summarize_symmetry_groups(
    dataframe: pd.DataFrame,
) -> dict:
    """
    Summarize symmetry-aware structure grouping.

    Returns
    -------
    dict
        Total structures and number of unique symmetry groups.
    """

    if "symmetry_group" not in dataframe.columns:
        raise ValueError(
            "DataFrame must contain 'symmetry_group'."
        )

    return {
        "total_structures":
            len(dataframe),

        "unique_structure_groups":
            int(
                dataframe[
                    "symmetry_group"
                ].nunique()
            ),
    }
