"""NSCool reference cases for the C2 physics tests and analysis.

``dump(mass, gaps, eta)`` caches ``NSCool.zone_dump`` on the test
temperatures; ``gate_error`` is the C2 comparison: the worst relative
error where a term is at least 1e-6 of the total, and elsewhere the worst
absolute error over the total.
"""
from __future__ import annotations

import functools

import numpy as np

from ..runner import GapModel, NSCool

__all__ = ["GAPS", "T_TEST", "dump", "gate_error"]

#: gap sets of C1 (nmb-papers analyses/c1_isothermal/run_case.py), plus one
#: that keeps protons paired in the dUrca core of APR 2.0 (two-gap suppression)
GAPS = {
    "none": GapModel(n1s0=0, n3p2=0, p1s0=0),
    "default": GapModel(),
    "zhou": GapModel(sx_p1s0=0.8, sy_p1s0=0.4),
    "zhou_no3p2": GapModel(n3p2=0, sx_p1s0=0.8, sy_p1s0=0.4),
    "c_ccdk": GapModel(n3p2=103, p1s0=7, sx_p1s0=2.0),
    # the remaining supported codes, with both scalings on n3P2
    "ccdk_n": GapModel(n1s0=2, n3p2=102, p1s0=7, sx_n3p2=1.3, sy_n3p2=0.7, sx_p1s0=1.5),
}

#: redshifted temperatures of the gates, 1e7 to 1e10 K
T_TEST = tuple(np.logspace(7, 10, 13))


@functools.lru_cache(maxsize=None)
def dump(mass, gaps="default", eta=0.0, eos_table=None, T=T_TEST, rho_b=1.6e14):
    """Zone dump with the scan on the same T grid (so ``iso`` rows line up)."""
    return NSCool().zone_dump(list(T), mass=mass, gaps=GAPS[gaps], eta=eta,
                              eos_table=eos_table, rho_b=rho_b,
                              T_grid=(T[0], T[-1], len(T)))


def gate_error(val, ref, total, floor=1e-6):
    """(worst relative error where |ref| >= floor * total, worst |val - ref| / total elsewhere)."""
    val, ref, total = (np.asarray(x, float) for x in (val, ref, total))
    big = (np.abs(ref) >= floor * total) & (total > 0)
    rel = np.max(np.abs(val[big] / ref[big] - 1.0), initial=0.0)
    tot = np.where(total > 0, total, 1.0)
    small = np.max((np.abs(val - ref) / tot)[~big], initial=0.0)
    return rel, small
