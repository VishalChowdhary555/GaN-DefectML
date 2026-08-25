"""
Defect structure generation utilities for GaN-DefectML.

Includes intrinsic defects, substitutional dopants, interstitials,
and defect-library construction.
"""

from .intrinsic_defects import (
    generate_ga_vacancy,
    generate_n_vacancy,
    generate_ga_on_n_antisite,
    generate_n_on_ga_antisite,
    generate_intrinsic_defects,
)

from .dopants import (
    generate_substitutional_dopant,
    generate_dopant_structures,
)

from .interstitials import (
    generate_interstitial_structure,
    generate_interstitial_structures,
    find_interstitial_candidates,
)

from .library import (
    build_master_structure_library,
    validate_master_structure_library,
)

__all__ = [
    "generate_ga_vacancy",
    "generate_n_vacancy",
    "generate_ga_on_n_antisite",
    "generate_n_on_ga_antisite",
    "generate_intrinsic_defects",
    "generate_substitutional_dopant",
    "generate_dopant_structures",
    "generate_interstitial_structure",
    "generate_interstitial_structures",
    "find_interstitial_candidates",
    "build_master_structure_library",
    "validate_master_structure_library",
]
