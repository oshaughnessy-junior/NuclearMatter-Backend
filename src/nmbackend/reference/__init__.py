"""Independent reference solvers for cross-validation (not in the hot path).

* ``scipy_tov`` : in-process radius-domain SciPy RK45 + (H, beta) tidal solver.
* ``reprimand`` : adapter to the external RePrimAnd high-accuracy TOV solver
                  (optional; requires ``pyreprimand``).
"""
from .scipy_tov import (solve_reference, reference_sequence, RefPoint,
                        solve_reference_hybrid)

__all__ = ["solve_reference", "reference_sequence", "RefPoint",
           "solve_reference_hybrid"]
