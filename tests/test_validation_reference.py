"""Cross-validate the production solver against an independent reference.

The reference (`nmbackend.reference.scipy_tov`) integrates in the *radius* domain
with SciPy adaptive RK45 and the (H, beta) tidal system -- a different variable,
stepper, and tidal formulation from the production enthalpy/diffrax solver.  Close
agreement corroborates both.

This is the test that caught the original fixed-step under-resolution (M off ~5%,
Lambda off ~25%); the adaptive diffrax backend now agrees to <0.1%.
"""
import numpy as np
import pytest

from nmbackend.eos import MetaModelEOS, PolytropeEOS
from nmbackend.tov.solver import solve_structure, _HAVE_DIFFRAX
from nmbackend.reference.scipy_tov import solve_reference

pytestmark = pytest.mark.skipif(not _HAVE_DIFFRAX,
                                reason="diffrax backend required for this test")


@pytest.mark.parametrize("eos", [MetaModelEOS(), PolytropeEOS()])
def test_diffrax_matches_reference(eos):
    t = eos.eos_table()
    for frac in (0.10, 0.20, 0.35, 0.55, 0.75):
        hc = frac * float(t.h[-1])
        ours = solve_structure(t, hc, backend="diffrax", rtol=1e-8, atol=1e-10)
        ref = solve_reference(t, hc, rtol=1e-9, atol=1e-11)
        assert abs(float(ours.M_msun) - ref.M_msun) <= 2e-3 * ref.M_msun
        assert abs(float(ours.R_km) - ref.R_km) <= 2e-3 * ref.R_km
        assert abs(float(ours.Lambda) - ref.Lambda) <= 5e-3 * abs(ref.Lambda)


def test_scan_converges_to_reference():
    """The fixed-step scan must converge to the reference as n_steps grows."""
    t = MetaModelEOS().eos_table()
    hc = 0.20 * float(t.h[-1])
    ref = solve_reference(t, hc, rtol=1e-9, atol=1e-11)
    errs = []
    for ns in (2000, 8000):
        m = float(solve_structure(t, hc, backend="scan", n_steps=ns).M_msun)
        errs.append(abs(m - ref.M_msun) / ref.M_msun)
    assert errs[1] < errs[0]            # error decreases with resolution
