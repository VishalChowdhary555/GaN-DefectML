"""
Crystal-structure analysis utilities for GaN-DefectML.

Provides host selection, supercell construction, structural
characterization, symmetry analysis, and local-environment analysis.
"""

from .host_selection import (
    select_primary_host,
    validate_primary_host,
)

from .supercell import (
    build_supercell,
    summarize_supercell,
    get_species_indices,
    select_representative_site,
)

from .symmetry import (
    analyze_symmetry,
    validate_symmetry,
)

from .geometry import (
    extract_lattice_parameters,
    calculate_volume_per_atom,
    calculate_periodic_distance,
)

from .neighbors import (
    get_neighbor_environment,
    calculate_neighbor_statistics,
    calculate_species_neighbor_counts,
)

__all__ = [
    "select_primary_host",
    "validate_primary_host",
    "build_supercell",
    "summarize_supercell",
    "get_species_indices",
    "select_representative_site",
    "analyze_symmetry",
    "validate_symmetry",
    "extract_lattice_parameters",
    "calculate_volume_per_atom",
    "calculate_periodic_distance",
    "get_neighbor_environment",
    "calculate_neighbor_statistics",
    "calculate_species_neighbor_counts",
]
