"""JAX gap Tc(k_F) and splines (nmbackend.cooling.physics.gaps) against
NSCool's get_Tc, zone by zone.  NSCool tests skip unless it is built;
NSCOOL_REQUIRED=1 (CI) makes them run."""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from nmbackend.cooling import GapModel, default_home
from nmbackend.cooling.physics import gaps as G
from nmbackend.cooling.physics._spline import natural_y2_matrix, spline_y2, splint

built = pytest.mark.skipif(
    not (default_home() / "Code" / "NSCool_iso.out").is_file()
    and not os.environ.get("NSCOOL_REQUIRED"),
    reason="NSCool_iso.out not built (nmbackend/cooling/build_nscool.sh)")


@pytest.mark.parametrize("bc", ["natural", "clamped"])
def test_spline_matches_scipy(bc):
    x = np.array([0.0, 0.3, 0.41, 0.6, 0.9, 1.3])
    y = np.sin(3 * x) + x ** 2
    if bc == "natural":
        y2, ref = spline_y2(x, y), CubicSpline(x, y, bc_type="natural")
    else:
        y2, ref = spline_y2(x, y, 0.0, 0.0), CubicSpline(x, y, bc_type="clamped")
    xx = np.linspace(-0.2, 1.5, 77)          # NR extrapolates the end cubics
    np.testing.assert_allclose(splint(x, y, y2, xx), ref(xx), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(natural_y2_matrix(x) @ y, spline_y2(x, y), atol=1e-12)


def test_tc_fits_shape():
    """Peaks where the tables put them, zero outside, finite k_F gradient."""
    k = np.linspace(0.0, 3.5, 701)
    sfb = np.asarray(G.tc_n1s0(1, k))
    assert k[np.argmax(sfb)] == pytest.approx(0.85, abs=0.05)
    assert sfb[k >= 1.45].max() == 0.0 and sfb[k <= 0.1].max() == 0.0
    assert np.asarray(G.tc_p1s0(3, k)).max() == pytest.approx(3.3e9, rel=0.01)
    a = G.tc_n3p2_gauss(k, *G.N3P2_GAUSS[101])
    assert k[np.argmax(a)] == pytest.approx(1.8, abs=0.005)
    d = jax.vmap(jax.grad(lambda x: G.tc_p1s0(3, x)))(k)
    assert np.all(np.isfinite(d))


@built
@pytest.mark.parametrize("mass", [1.4, 2.0])
@pytest.mark.parametrize("gaps", ["none", "default", "zhou", "zhou_no3p2", "c_ccdk",
                                  "ccdk_n"])
def test_star_tc_matches_nscool(mass, gaps):
    from nmbackend.cooling.physics import reference as R
    d = R.dump(mass, gaps, T=(1e8,))
    st, n = d.star, d.imax + 1
    tc = G.star_tc(R.GAPS[gaps], st["kfn"], st["kfp"], st["nb"], d.icore, d.idrip)
    assert int(tc.isf) == d.isf
    np.testing.assert_allclose(np.asarray(tc.tcn)[:n], st["Tcn"][:n], rtol=2e-6)
    np.testing.assert_allclose(np.asarray(tc.tcp)[:n], st["Tcp"][:n], rtol=2e-6)


@built
def test_sx_moves_the_gap_and_is_differentiable():
    """s_x < 1 evaluates the gap at a lower density; d Tc / d s_x is finite."""
    from nmbackend.cooling.physics import reference as R
    d = R.dump(1.4, "default", T=(1e8,))
    st = d.star
    f = lambda sx: G.star_tc(GapModel(), st["kfn"], st["kfp"], st["nb"], d.icore,
                             d.idrip, sx_p1s0=sx).tcp
    g = jax.jacfwd(f)(0.8)
    h = 1e-6
    fd = (np.asarray(f(0.8 + h)) - np.asarray(f(0.8 - h))) / (2 * h)
    np.testing.assert_allclose(np.asarray(g), fd, rtol=1e-4, atol=1e-6 * np.abs(fd).max())
    assert np.abs(fd).max() > 0


def test_below_keeps_gradients_finite():
    """tau stays below 1 at T = Tc, and Tc = 0 counts as unpaired."""
    from nmbackend.cooling.physics.superfluid import below, u_1s0
    f = lambda t, tc, strict: jnp.sum(jnp.where(below(t, tc, strict)[0],
                                                u_1s0(below(t, tc, strict)[1]) ** 2, 0.0))
    for strict in (True, False):
        for tc in (1e9, 0.0):
            g = jax.grad(f, argnums=(0, 1))(1e9, tc, strict)
            assert np.all(np.isfinite(g))


def test_non_integer_gap_code_is_refused():
    with pytest.raises(ValueError):
        G.star_tc(GapModel(n1s0=1.5), np.ones(3), np.ones(3), np.ones(3), 1, 2)
