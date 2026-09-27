"""Rejection of cores whose raw pressure falls with density below the
maximum-mass star (nmbackend.eos.crust.monotone_core, finish_table)."""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nmbackend.eos import (DD2, SLY4, DDRMFEOS, MetaModelEOS, RMFTEOS,
                           SkyrmeEOS)
from nmbackend.eos.crust import (InvalidCoreError, monotone_core,
                                 unstable_below_mmax)
from nmbackend.tov import build_family, default_hc_grid

jax.config.update("jax_enable_x64", True)

# rmfwr prior draw (paper6 pilot 2, seed 1, draw 139): the raw pressure falls
# above n = 0.93 fm^-3, beyond the maximum-mass central density 0.67 fm^-3
RMF_DIP_ABOVE_MMAX = dict(M_mev=939.0, C_sigma=15.680789381761235,
                          C_omega=10.654742346294759, C_rho=17.569217414705925,
                          b=0.002037324482721107, c=-0.002946206687004775,
                          Lambda_v=0.057087171139297656)


def test_monotone_core_matches_cummax():
    """p is floor + cummax, bit for bit; n_u is where the drop begins."""
    n = jnp.linspace(0.1, 1.0, 10)
    p = jnp.array([-1.0, 1.0, 2.0, 3.0, 2.5, 2.0, 3.5, 4.0, 5.0, 6.0])
    pm, lnu = monotone_core(n, p)
    np.testing.assert_array_equal(pm, jax.lax.cummax(jnp.maximum(p, 1e-9)))
    assert float(lnu) == pytest.approx(math.log(0.4))
    _, lnu = monotone_core(n, jnp.arange(1.0, 11.0))
    assert np.isinf(float(lnu))


def test_unstable_metamodel_rejected():
    """L = 20, K_sym = -400: the raw pressure falls from n = 0.16 fm^-3 to
    n_match.  Error when eager, NaN table under jit, plateau table with the
    escape hatch."""
    P = {"L_sym": 20.0, "K_sym": -400.0}
    with pytest.raises(InvalidCoreError):
        MetaModelEOS().eos_table(P)
    t = jax.jit(MetaModelEOS().eos_table)(P)
    assert np.all(np.isnan(np.asarray(t.p)))
    t = MetaModelEOS(allow_unstable_core=True).eos_table(P)
    p = np.asarray(t.p)
    assert np.all(np.isfinite(p)) and np.sum(np.diff(p) == 0.0) > 50


@pytest.mark.parametrize("cls, par", [
    (MetaModelEOS, None), (lambda **k: MetaModelEOS(muons=True, **k), None),
    (RMFTEOS, None), (SkyrmeEOS, SLY4), (DDRMFEOS, DD2),
    (lambda **k: RMFTEOS(crust="polytrope", **k), None)])
def test_default_tables_unchanged(cls, par):
    """Stable cores: the table is bit-identical to the unchecked one."""
    t = cls().eos_table(par)
    u = cls(allow_unstable_core=True).eos_table(par)
    for a, b in zip(t, u):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_dip_above_mmax_kept():
    """An RMF core that softens only above the maximum-mass central density
    is kept; the same table would be rejected if the dip began at 0.5 fm^-3."""
    t = RMFTEOS().eos_table(RMF_DIP_ABOVE_MMAX)
    assert np.all(np.isfinite(np.asarray(t.p)))
    assert bool(unstable_below_mmax(t, math.log(0.5)))
    assert not bool(unstable_below_mmax(t, math.log(0.95)))


def test_lambda_smooth_to_rejection_boundary():
    """L = 20: bisect K_sym to the rejection boundary; Lambda(1.4) on the
    accepted side matches a quadratic trend from farther out within 1e-3
    (a plateau allowance of 1% in n gave 6e-3)."""
    mm = MetaModelEOS()

    def ok(K):
        try:
            mm.eos_table({"L_sym": 20.0, "K_sym": K})
            return True
        except InvalidCoreError:
            return False

    lo, hi = -190.0, -170.0
    assert not ok(lo) and ok(hi)
    for _ in range(8):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if ok(mid) else (mid, hi)
    fam = jax.jit(lambda t: build_family(t, default_hc_grid(t, n_models=120)))

    def lam14(K):
        f = fam(mm.eos_table({"L_sym": 20.0, "K_sym": K}))
        M, L, st = (np.asarray(a) for a in (f.M_msun, f.Lambda, f.stable))
        b = np.flatnonzero(st)
        b = b[b <= b[np.argmax(M[b])]]
        return float(np.exp(np.interp(1.4, M[b], np.log(L[b]))))

    d = np.array([0.1, 0.3, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    lam = np.array([lam14(hi + x) for x in d])
    assert np.all(np.diff(lam) > 0)
    c = np.polyfit(d[2:], lam[2:], 2)
    np.testing.assert_allclose(lam, np.polyval(c, d), rtol=1e-3)
