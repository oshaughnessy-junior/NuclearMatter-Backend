"""Curve grid (nmbackend.cooling.grid): the factored evaluation equals the
plain JAX isothermal curve, and the EOS-level grid has the documented layout.
Skips unless NSCool is built; NSCOOL_REQUIRED=1 (CI) makes it run."""
import os

import numpy as np
import pytest

from nmbackend.cooling import GapModel, default_home

pytestmark = pytest.mark.skipif(
    not (default_home() / "Code" / "NSCool_iso.out").is_file()
    and not os.environ.get("NSCOOL_REQUIRED"),
    reason="NSCool_iso.out not built (nmbackend/cooling/build_nscool.sh)")

from nmbackend.cooling.grid import _on_ages, eos_curve_grid, star_curves  # noqa: E402
from nmbackend.cooling.physics import Star  # noqa: E402
from nmbackend.cooling.physics import reference as R  # noqa: E402
from nmbackend.cooling.physics.curve import isothermal_curve  # noqa: E402

T_AGE = np.geomspace(1e2, 3e6, 31)
GAPS = [R.GAPS["default"], R.GAPS["zhou"], R.GAPS["c_ccdk"]]


def test_factored_curves_match_plain_curve():
    """Shared sums, deduplicated boundary gaps and the coarse crust march
    reproduce physics.curve.isothermal_curve(blanket=True) on the same grid."""
    s = Star.from_dump(R.dump(2.0))
    T = np.geomspace(3e4, 1e10, 400)
    eta = np.array([0.0, 1e-9])
    c = star_curves(s, GAPS, eta, T_AGE, T_grid=T)
    assert c.logL.shape == (3, 2, T_AGE.size)
    for i, g in enumerate(GAPS):
        for j, e in enumerate(eta):
            p = isothermal_curve(T, s, s.tc(g), eta=float(e), rho_b=s.rhocore)
            ref, v, _ = _on_ages(np.asarray(p.t_yr), np.asarray(p.L_gamma_inf), T_AGE)
            np.testing.assert_array_equal(v, c.valid[i, j])
            assert np.nanmax(np.abs(ref - c.logL[i, j])) < 2e-3


def test_eos_curve_grid_layout():
    """Bundled APR: two buildable masses, one past NSCool's M_max (flagged)."""
    f = np.array([0.0, 1e-8])
    o = eos_curve_grid(None, [1.4, 2.0, 2.5], GAPS[:1], f, T_AGE,
                       T_grid=np.geomspace(3e4, 1e10, 200))
    assert o["logL"].shape == (3, 1, 2, T_AGE.size)
    assert list(o["star_ok"]) == [True, True, False] and o["error"][2]
    assert np.all(np.isnan(o["logL"][2])) and not o["valid"][2].any()
    np.testing.assert_allclose(o["eta_pcy"][:2, 1], o["gs14"][:2] ** 2 * 1e-8)
    assert o["n_durca_zones"][0] == 0 and o["n_durca_zones"][1] > 0   # APR M_DU ~ 2.0
    ok = o["valid"][:2]
    assert ok[..., T_AGE <= 1e5].all()
    # light elements are brighter at fixed age while neutrinos dominate
    young = T_AGE <= 1e4
    assert np.all(o["logL"][:2, 0, 1, young] > o["logL"][:2, 0, 0, young])
    assert np.all(np.diff(o["logL"][:2, 0, 0, young], axis=-1) < 0)
