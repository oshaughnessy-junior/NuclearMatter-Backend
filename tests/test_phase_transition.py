"""First-order phase-transition (hybrid hadron-quark) EOS + tidal junction.

The decisive check: the production enthalpy solver with the analytic tidal junction
(`tov.phase_transition.solve_hybrid`) vs an *independent* radius-domain solver with
the junction in the (H,beta) formulation (`reference.scipy_tov.solve_reference_hybrid`).
Agreement to <0.5% corroborates the junction code across two formulations.  (The
external arbiter remains RePrimAnd; see docs/environment.md.)
"""
import numpy as np
import pytest

from nmbackend.eos import HybridEOS
from nmbackend.tov.solver import _HAVE_DIFFRAX
from nmbackend.tov.phase_transition import solve_hybrid
from nmbackend.reference.scipy_tov import solve_reference_hybrid

pytestmark = pytest.mark.skipif(not _HAVE_DIFFRAX, reason="diffrax required")


def _hybrid():
    return HybridEOS(p_trans_mev=80.0, delta_eps_mev=250.0, cs2_quark=1.0)


def test_transition_metadata_sane():
    hy = _hybrid()
    tr = hy.transition
    assert tr.eps_in > tr.eps_out > 0.0     # density jumps up going inward
    assert 0.0 < tr.h_t < float(hy.segments[0].h[-1])  # within inner segment range


def test_hybrid_sequence_physical():
    hy = _hybrid()
    tr = hy.transition
    # below transition: purely hadronic; above: quark core
    lo = solve_hybrid(hy, 0.7 * tr.h_t)
    hi = solve_hybrid(hy, 1.8 * tr.h_t)
    assert float(hi.M_msun) > float(lo.M_msun)        # heavier with quark core
    assert float(hi.R_km) < float(lo.R_km)            # transition softens -> smaller R
    assert 0.0 < float(hi.Lambda) < float(lo.Lambda)  # and smaller Lambda


def test_junction_matches_independent_reference():
    hy = _hybrid()
    tr = hy.transition
    for fr in (0.7, 1.4, 2.0):                         # hadronic + two quark-core
        hc = fr * tr.h_t
        a = solve_hybrid(hy, hc, rtol=1e-9, atol=1e-11)
        b = solve_reference_hybrid(hy, hc, rtol=1e-10, atol=1e-12)
        assert abs(float(a.M_msun) - b.M_msun) <= 5e-3 * b.M_msun
        assert abs(float(a.R_km) - b.R_km) <= 5e-3 * b.R_km
        assert abs(float(a.Lambda) - b.Lambda) <= 5e-3 * abs(b.Lambda)


def test_hadronic_only_branch_matches_plain_solver():
    from nmbackend.tov.solver import solve_structure
    hy = _hybrid()
    tr = hy.transition
    _, outer = hy.segments
    hc = 0.6 * tr.h_t
    a = solve_hybrid(hy, hc)
    b = solve_structure(outer, hc, backend="diffrax")
    assert abs(float(a.Lambda) - float(b.Lambda)) <= 1e-6 * float(b.Lambda)
