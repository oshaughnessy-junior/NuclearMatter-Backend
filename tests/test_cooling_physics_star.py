"""JAX crust conductivity, envelope, isothermal sums and curve
(nmbackend.cooling.physics) against NSCool and the C1 driver, and AD
against finite differences.  Skips unless NSCool is built;
NSCOOL_REQUIRED=1 (CI) makes it run."""
import dataclasses
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nmbackend.cooling import default_home, isothermal_curve as c1_curve

pytestmark = pytest.mark.skipif(
    not (default_home() / "Code" / "NSCool_iso.out").is_file()
    and not os.environ.get("NSCOOL_REQUIRED"),
    reason="NSCool_iso.out not built (nmbackend/cooling/build_nscool.sh)")

from nmbackend.cooling.physics import conduct as K  # noqa: E402
from nmbackend.cooling.physics import envelope as E  # noqa: E402
from nmbackend.cooling.physics import reference as R  # noqa: E402
from nmbackend.cooling.physics.curve import isothermal_curve  # noqa: E402
from nmbackend.cooling.physics.star import Star, isothermal_sums, zone_terms  # noqa: E402


@pytest.mark.parametrize("mass", [1.4, 2.0])
def test_crust_conductivity_matches_nscool(mass):
    d = R.dump(mass)
    cr = d.zones[d.zones > d.icore]
    s = {k: d.star[k][cr] for k in ("rho", "A_cell", "Z_ion", "kfe")}
    T = d.terms["T"][:, -cr.size:] / d.star["ephi"][cr]
    lam = jax.jit(K.lambda_crust)(T, s["rho"], s["A_cell"], s["Z_ion"], s["kfe"],
                                  d.qimp, d.rhodrip)
    np.testing.assert_allclose(lam, d.terms["lam"][:, -cr.size:], rtol=1e-6)
    gamma = 2.273e5 * s["Z_ion"] ** 2 * (s["rho"] / s["A_cell"]) ** (1 / 3) / T
    assert (gamma > 210).any() and (gamma < 210).any()        # solid and liquid


def test_envelope_is_pcy97():
    """PCY97 A7-A10 with PCY97's eta (NSCool's ETA / 0.443): iron at eta = 0,
    hotter with light elements, and the mix of A10."""
    tb, g = np.logspace(7, 9.5, 11), 1.8
    t9 = tb / 1e9
    z = t9 - np.sqrt(7 * t9 * np.sqrt(g)) / 1e3
    t4_fe = g * ((7 * z) ** 2.25 + (z / 3) ** 1.25)
    t4_acc = g * (18.1 * t9) ** 2.42
    np.testing.assert_allclose(E.teff_local(tb, 0.0, g), t4_fe ** 0.25 * 1e6, rtol=1e-12)
    for eta in (1e-13, 1e-9, 1e-7):
        a = (1.2 + (5.3e-6 / eta) ** 0.38) * t9 ** (5 / 3)
        mix = ((a * t4_fe + t4_acc) / (a + 1)) ** 0.25 * 1e6
        np.testing.assert_allclose(E.teff_local(tb, eta, g), mix, rtol=1e-6)
        assert np.all(np.asarray(E.teff_local(tb, eta, g)) > t4_fe ** 0.25 * 1e6)


@pytest.mark.parametrize("mass,gaps,eta", [(1.4, "default", 1e-9), (2.0, "none", 0.0),
                                           (2.0, "zhou", 1e-9)])
def test_isothermal_sums_match_nscool(mass, gaps, eta):
    d = R.dump(mass, gaps, eta)
    s = Star.from_dump(d)
    tc = s.tc(R.GAPS[gaps])
    f = jax.jit(jax.vmap(lambda T: isothermal_sums(T, s, tc, d.eta_nscool if eta else 0.0,
                                                   rho_b=1.6e14)))
    r = f(jnp.asarray(d.T))
    for k in ("C", "L_nu", "L_gamma", "Teff_inf", "C_core", "L_nu_crust"):
        np.testing.assert_allclose(getattr(r, k), getattr(d.iso, k), rtol=1e-6, err_msg=k)
    np.testing.assert_allclose(r.L_durca, d.iso.L_durca, rtol=1e-6, atol=1e-300)
    # NSCool's last march step uses its core conductivity at zone icore (C4)
    np.testing.assert_allclose(r.T_in, d.iso.T_in, rtol=2e-3)


def test_march_is_exact_inside_the_crust():
    d = R.dump(1.4, "default", 1e-9, rho_b=1e13)
    s = Star.from_dump(d)
    r = jax.jit(jax.vmap(lambda T: isothermal_sums(T, s, s.tc(R.GAPS["default"]),
                                                   d.eta_nscool, rho_b=1e13)))(jnp.asarray(d.T))
    np.testing.assert_allclose(r.T_in, d.iso.T_in, rtol=1e-7)
    assert np.all(d.iso.T_in[-4:] > 1.03 * np.asarray(d.T[-4:]))   # the march does work


def test_curve_matches_c1_driver():
    """t >= 1 kyr within 1e-3 dex (gate: 0.01) on the C1 grid, APR 2.0, no gaps."""
    d = R.dump(2.0, "none", 1e-9, T=tuple(np.logspace(4, 10, 481)))
    ref = c1_curve(d.iso, blanket=True)
    s = Star.from_dump(d)
    c = jax.jit(lambda: isothermal_curve(np.logspace(4, 10, 481), s,
                                         s.tc(R.GAPS["none"]), d.eta_nscool, 1.6e14))()
    ages = np.logspace(3, 6, 31)
    lg = lambda u: np.interp(np.log(ages), np.log(np.asarray(u.t_yr)),
                             np.log10(np.asarray(u.Teff_inf)))
    assert np.max(np.abs(lg(c) - lg(ref))) < 1e-3


def _fd_check(f, x0, rev=True, rtol=1e-5):
    fj = jax.jit(f)
    n = len(x0)
    jf = jax.jit(jax.jacfwd(f, argnums=tuple(range(n))))(*x0)
    jr = jax.jit(jax.jacrev(f, argnums=tuple(range(n))))(*x0) if rev else jf
    for k in range(n):
        h = 1e-5
        xp, xm = list(x0), list(x0)
        xp[k] += h
        xm[k] -= h
        fd = (np.asarray(fj(*xp)) - np.asarray(fj(*xm))) / (2 * h)
        np.testing.assert_allclose(jf[k], fd, rtol=rtol, atol=1e-6, err_msg=str(k))
        np.testing.assert_allclose(jr[k], jf[k], rtol=1e-10, atol=1e-12, err_msg=str(k))


def test_gradients_match_finite_differences():
    """d/d(log T, s_y, s_x, k_Fp scale, m*_n scale, log eta) of log C, L_nu,
    L_gamma (APR 2.0, dUrca zone); forward and reverse mode agree."""
    from nmbackend.cooling import GapModel
    d = R.dump(2.0, "default", 1e-9)
    s0 = Star.from_dump(d)

    def f(lT, sy, sx, kp, mn, leta):
        s = dataclasses.replace(s0, kfp=s0.kfp * kp, mstn=s0.mstn * mn)
        r = isothermal_sums(10.0 ** lT, s, s.tc(GapModel(), sy_p1s0=sy, sx_p1s0=sx),
                            10.0 ** leta)
        return jnp.log(jnp.stack([r.C, r.L_nu, r.L_gamma]))

    _fd_check(f, (8.3, 0.9, 1.1, 1.0, 1.0, -8.6))


def test_march_gradient():
    """d log T_in / d(log T, log eta) through the crust march."""
    d = R.dump(1.4, "default", 1e-9)
    s = Star.from_dump(d)
    tc = s.tc(R.GAPS["default"])
    f = lambda lT, leta: jnp.log(isothermal_sums(10.0 ** lT, s, tc, 10.0 ** leta,
                                                 rho_b=1e12).T_in)
    _fd_check(f, (8.7, -8.6))


@pytest.mark.parametrize("mass,gaps", [(1.4, "default"), (2.0, "none"), (2.0, "zhou"),
                                       (1.4, "c_ccdk")])
def test_zone_terms_match_dump(mass, gaps):
    """The assembled star: every term per zone, with JAX Tc and NSCool's masks."""
    d = R.dump(mass, gaps)
    s = Star.from_dump(d)
    t = jax.jit(jax.vmap(lambda T: zone_terms(T, s, s.tc(R.GAPS[gaps]),
                                              R.GAPS[gaps].ca_n3p2)))(jnp.asarray(d.T))
    crust = d.zones > d.icore
    for k in set(t) - {"T"}:
        tot = d.terms["qnu" if k.startswith("q") else "cv"]
        if k == "lam":
            np.testing.assert_allclose(t[k][:, crust], d.terms[k][:, crust], rtol=1e-6)
            continue
        rel, small = R.gate_error(t[k], d.terms[k], tot)
        assert rel < 1e-5 and small < 1e-8, (k, rel, small)


def test_get_lam_reverse_mode_is_finite():
    """Large w (T >~ 1e10 K) must not leak exp(s w) overflow into gradients."""
    from nmbackend.cooling.physics.conduct import _get_lam
    for s_, w in ((0.01, 800.0), (1.0, 1e3), (0.01, 1e5), (0.3, 5.0)):
        g = jax.grad(lambda a, b: sum(_get_lam(a, b)), argnums=(0, 1))(s_, w)
        assert np.all(np.isfinite(g)), (s_, w)


def test_march_refuses_rho_b_in_the_core():
    d = R.dump(1.4)
    s = Star.from_dump(d)
    with pytest.raises(ValueError):
        isothermal_sums(1e8, s, s.tc(R.GAPS["default"]), rho_b=5e14)
