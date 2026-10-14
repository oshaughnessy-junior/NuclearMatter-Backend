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


def test_junction_matches_smeared_transition():
    """Sign of the tidal junction.  The sharp transition with the junction must be
    the limit of the same transition smeared over a pressure width w and
    integrated by the plain solver (continuous tidal term).  On HybridEOS() at
    h_c = h_t + 0.08 the smeared Lambda converges linearly in w: -3.8e-4,
    -7.6e-5, -1.9e-5 at w = 1e-3, 2e-4, 5e-5 against the junction's 97.726.
    With the junction sign of before 2026-10-14 the hybrid gave 95.60 (-2.2%).
    (The smeared table no longer adds 1e-12 k MeV/fm^3 to every pressure,
    which left a +2e-4 floor.)"""
    from nmbackend.tov.solver import solve_structure
    hy = HybridEOS()
    hc = float(hy.transition.h_t) + 0.08
    sm = solve_structure(hy.eos_table_smeared(width_frac=5e-5), hc,
                         rtol=1e-10, atol=1e-12)
    hyb = solve_hybrid(hy, hc, rtol=1e-10, atol=1e-12)
    assert abs(float(hyb.M_msun) / float(sm.M_msun) - 1.0) < 1e-5
    assert abs(float(hyb.Lambda) / float(sm.Lambda) - 1.0) < 1e-4


def test_nudged_plateau_is_continuous():
    """A density jump at fixed p written into one table (the raw combined
    HybridEOS table), with the quark side nudged by delta p_t to make p strictly
    increasing.  Lambda from the plain solver, and from the consistent_cs2
    reference, must stay at the junction value of solve_hybrid for every delta.
    Before the chord test for jump cells, a cell between 1e3 ulp and ~3e-11 wide
    lost most of its Delta eps to the 1e10 (eps + p) clip (delta 1e-12..1e-10:
    -1.2e-2), and the reference skipped thin segments (delta 1e-11..3e-8:
    -1.24e-2).  Measured now: solver +1.43e-5 (junction bubble) up to delta 1e-8,
    -1e-8 above; reference <= 1.1e-6."""
    import math
    import jax
    import jax.numpy as jnp
    from nmbackend.eos.base import build_h_and_cs2
    from nmbackend.reference.scipy_tov import solve_reference
    from nmbackend.tov.solver import solve_structure
    hy = HybridEOS(n_hadronic=None)
    hc = float(hy.transition.h_t) + 0.08
    L = float(solve_hybrid(hy, hc, rtol=1e-10, atol=1e-12).Lambda)
    no, eo, po = hy._had_seg
    nq, eq, pq = hy._quark_seg
    pt = float(hy.p_trans_mev)
    lam = jax.jit(lambda t: solve_structure(t, hc, rtol=1e-10, atol=1e-12).Lambda)
    for d in (0.0, 1e-13, 1e-11, 1e-9, 3e-8, 1e-6, 1e-5):   # 1e-5 crosses the
        # reference's 1e-6 chord threshold (delta ~2.5e-6)
        t = build_h_and_cs2(np.concatenate([no, nq]), np.concatenate([eo, eq]),
                            np.concatenate([po, pq + d * pt]))
        assert abs(float(lam(t)) / L - 1.0) < 5e-5, d
        p_c = float(t.at_h(hc)[1])
        hr = np.interp(math.log(p_c), np.log(np.asarray(t.p)), np.asarray(t.h))
        ref = solve_reference(t, hr, rtol=1e-10, atol=1e-40, consistent_cs2=True)
        assert abs(ref.Lambda / L - 1.0) < 5e-5, d
