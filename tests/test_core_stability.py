"""Rejection of cores whose raw pressure falls with density below the
maximum-mass star (nmbackend.eos.crust.monotone_core, finish_table)."""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nmbackend.eos import (DD2, SLY4, DDRMFEOS, MetaModelEOS, RMFTEOS,
                           SkyrmeEOS)
from nmbackend.eos.crust import (InvalidCoreError, UnstableCoreError,
                                 monotone_core, unstable_below_mmax)
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
    with pytest.raises(UnstableCoreError):
        MetaModelEOS().eos_table(P)
    t = jax.jit(MetaModelEOS().eos_table)(P)
    assert np.all(np.isnan(np.asarray(t.p)))
    t = MetaModelEOS(allow_unstable_core=True).eos_table(P)
    p = np.asarray(t.p)
    assert np.all(np.isfinite(p)) and np.sum(np.diff(p) == 0.0) > 50


def test_crust_rejection_is_not_unstable_core():
    """A core rejected at the crust join raises the parent class only."""
    with pytest.raises(InvalidCoreError) as e:
        MetaModelEOS().eos_table({"L_sym": 20.0, "K_sym": 0.0})
    assert not isinstance(e.value, UnstableCoreError)


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


# review of PR #9: the raw p falls from 0.97 fm^-3 to the table top, where
# h = h_max; the check once crashed diffrax there (max_steps)
RMF_DIP_AT_TOP = dict(M_mev=939.0, C_sigma=12.616047358874976,
                      C_omega=8.560661106535438, C_rho=15.931769773614421,
                      b=0.003505489329273, c=0.0062758417594662875,
                      Lambda_v=0.04909735507220698)
# review of PR #9: no dip; jit(vmap) once solved a star at h_max and crashed
RMF_STABLE = dict(M_mev=939.0, C_sigma=12.3617, C_omega=6.4937,
                  C_rho=16.3866, b=0.007598, c=-0.006864, Lambda_v=0.041927)


def test_dip_at_table_top_kept():
    m = RMFTEOS()
    for f in (m.eos_table, jax.jit(m.eos_table)):
        assert np.all(np.isfinite(np.asarray(f(RMF_DIP_AT_TOP).p)))


def test_vmap_batch():
    """jit(vmap): stable and top-dip cores stay finite, an unstable one is
    NaN, and a NaN element does not break the batch."""
    batch = [RMF_STABLE, RMF_DIP_AT_TOP, RMF_DIP_ABOVE_MMAX]
    X = {k: jnp.array([d[k] for d in batch]) for k in RMF_STABLE}
    p = np.asarray(jax.jit(jax.vmap(RMFTEOS().eos_table))(X).p)
    assert np.all(np.isfinite(p))
    mm = jax.jit(jax.vmap(MetaModelEOS().eos_table))
    p = np.asarray(mm({"L_sym": jnp.array([20.0, 60.0]),
                       "K_sym": jnp.array([-400.0, 0.0])}).p)
    assert np.all(np.isnan(p[0])) and np.all(np.isfinite(p[1]))


def test_nan_core_not_reported_unstable():
    """A NaN from elsewhere is returned as before (polytrope surface),
    not raised as UnstableCoreError."""
    from nmbackend.eos.crust import finish_table
    n = jnp.geomspace(0.05, 1.0, 40)
    p = (n / 0.16) ** 2 * 10.0
    p = p.at[20].set(jnp.nan)
    eps = 939.0 * n + p
    pm, lnu = monotone_core(n, p)
    t = finish_table(n, eps, pm, "polytrope", 0.055, 1e-7, 1.3, lnu)
    assert not np.all(np.isfinite(np.asarray(t.p)))
