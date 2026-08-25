"""
Materials data collection utilities for GaN-DefectML.

Provides Materials Project retrieval, filtering, cleaning,
and dataset construction for Ga-N host structures.
"""

from .materials_project import (
    fetch_gan_entries,
    fetch_material_structure,
    query_materials_project,
)

from .preprocessing import (
    clean_materials_dataframe,
    filter_gan_entries,
    classify_stability,
    build_materials_dataset,
)

__all__ = [
    "query_materials_project",
    "fetch_gan_entries",
    "fetch_material_structure",
    "clean_materials_dataframe",
    "filter_gan_entries",
    "classify_stability",
    "build_materials_dataset",
]
