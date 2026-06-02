"""TOV solver and stellar-sequence tools."""
from .solver import solve_structure, TOVPoint
from .family import (
    build_family, Family, default_hc_grid, m_max, dhc_dM,
    lambda_of_mass, radius_of_mass,
)

__all__ = [
    "solve_structure", "TOVPoint",
    "build_family", "Family", "default_hc_grid", "m_max", "dhc_dM",
    "lambda_of_mass", "radius_of_mass",
]
