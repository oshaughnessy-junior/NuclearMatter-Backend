"""SLy crust join (nmbackend.eos.crust) and R_1.4 against published values."""
import functools

import jax
import numpy as np
import pytest

from nmbackend.eos import (DD2, SLY4, BSKG3, DDRMFEOS, MetaModelEOS, RMFTEOS,
                           SkyrmeEOS)
from nmbackend.eos.crust import N_T_DEFAULT, attach_crust, join_density
from nmbackend.tov import build_family, default_hc_grid

jax.config.update("jax_enable_x64", True)

MODELS = {"sly4": (SkyrmeEOS, SLY4), "bskg3": (SkyrmeEOS, BSKG3),
          "dd2": (DDRMFEOS, DD2), "gm1": (RMFTEOS, None)}


@functools.lru_cache(maxsize=None)
def star14(name, crust="sly", n_t=N_T_DEFAULT):
    """(R_1.4 [km], Lambda_1.4) on the stable branch."""
    cls, par = MODELS[name]
    t = cls(crust=crust, n_t=n_t).eos_table(par)
    f = build_family(t, default_hc_grid(t, n_models=60))
    M, R, L, st = (np.asarray(a) for a in (f.M_msun, f.R_km, f.Lambda, f.stable))
    b = np.flatnonzero(st)
    b = b[b <= b[np.argmax(M[b])]]
    return (float(np.interp(1.4, M[b], R[b])),
            float(np.exp(np.interp(1.4, M[b], np.log(L[b])))))


def test_sly4_matches_unified_sly():
    """Core and crust are both SLy4, so the unified SLy table is the target:
    R_1.4 = 11.736 km (Read et al. 2009, arXiv:0812.2163, Table III)."""
    assert abs(star14("sly4")[0] - 11.736) < 0.1


def test_bskg3_radius():
    """arXiv:2307.14276 Table 4: R_1.4 = 12.73 km (BSkG3 core, approximate
    crust).  Here the crust is SLy, not BSkG3."""
    assert abs(star14("bskg3")[0] - 12.73) < 0.15


def test_dd2_radius():
    """HS(DD2): R_1.4 = 13.22 km (Fischer et al. 2014, EPJA 50, 46,
    arXiv:1307.6190, Table 1; own non-SLy crust)."""
    assert abs(star14("dd2")[0] - 13.22) < 0.15


def test_gm1_radius():
    """Fortin et al. 2016 (arXiv:1604.01944) Table I, GM1 core glued to the
    DH crust at the crust-core transition: R_1.4 = 13.92 km (unified: 13.76)."""
    assert abs(star14("gm1")[0] - 13.92) < 0.1


def test_polytrope_flag_reproduces_old_tables():
    """crust="polytrope" is the previous default (R_1.4 = 11.36 km for SLy4)."""
    assert abs(star14("sly4", crust="polytrope")[0] - 11.36) < 0.01


def test_lambda_insensitive_to_crust():
    for name in ("sly4", "dd2"):
        a, b = star14(name)[1], star14(name, crust="polytrope")[1]
        assert abs(a / b - 1.0) < 0.01, name


@pytest.mark.parametrize("n_t", [0.0275, 0.11])
def test_transition_density_systematic_small(n_t):
    """SLy4 core on the SLy crust: the join density barely matters."""
    assert abs(star14("sly4", n_t=n_t)[0] - star14("sly4")[0]) < 0.03


@pytest.mark.parametrize("n_t, tol", [(0.0275, 0.1), (0.076, 0.15)])
def test_transition_density_stiff_metamodel(n_t, tol):
    """A stiff core (L = 110 MeV, K_sym = 0) on the SLy crust: R_1.4 is
    13.84 km at n_t = 0.055, +0.06 km at 0.0275 and -0.12 km at the SLy
    crust-core edge 0.076 fm^-3.  At 0.11 fm^-3 (-0.37 km) the 'crust'
    contains SLy4 core rows; not pinned."""
    p = dict(L_sym=110.0, K_sym=0.0)
    r = {}
    for nt in (N_T_DEFAULT, n_t):
        t = MetaModelEOS(muons=True, n_t=nt).eos_table(p)
        f = build_family(t, default_hc_grid(t, n_models=60))
        b = np.flatnonzero(np.asarray(f.stable))
        r[nt] = float(np.interp(1.4, np.asarray(f.M_msun)[b], np.asarray(f.R_km)[b]))
    assert abs(r[n_t] - r[N_T_DEFAULT]) < tol


def test_join_smooth_in_parameters():
    """Lambda at fixed h_c has a smooth slope in L_sym (no join-node jumps)."""
    from nmbackend.tov.solver import solve_structure
    mm = MetaModelEOS()
    Ls = np.linspace(59.7, 60.1, 9)
    hc = 0.2 * float(mm.eos_table({"L_sym": 60.0}).h[-1])
    lam = [float(solve_structure(mm.eos_table({"L_sym": L}), hc, rtol=1e-11,
                                 atol=1e-13).Lambda) for L in Ls]
    slope = np.diff(lam) / np.diff(Ls)
    assert np.ptp(slope) < 0.02 * np.abs(slope).mean()


def _is_fallback(params):
    from nmbackend.eos.crust import crust_nodes, GAMMA_MIN
    from nmbackend.eos.metamodel import DEFAULT_PARAMS
    _, ec, pc = crust_nodes()
    _, e, p = MetaModelEOS()._low_density(dict(DEFAULT_PARAMS, **params),
                                          n_min=join_density())
    return float(p[0]) < pc[-1] * (float(e[0]) / ec[-1]) ** GAMMA_MIN


@pytest.mark.parametrize("L, K_sym", [(60.0, 0.0), (80.0, 50.0)])
def test_fallback_join_smooth_and_differentiable(L, K_sym):
    """Cores softer than the crust at n_j (the join moves to n* > n_j):
    Lambda at fixed h_c is smooth in L_sym, and the fixed-mass IFT gradients
    of Lambda(1.4) and R(1.4) match a re-rooted finite difference."""
    from nmbackend.inference import sensitivity_matrix
    from nmbackend.tov.solver import solve_structure
    P = {"L_sym": L, "K_sym": K_sym}
    assert _is_fallback(P)
    mm = MetaModelEOS()
    hc = 0.2 * float(mm.eos_table(P).h[-1])
    Ls = L + 0.05 * np.arange(-3, 4)
    lam = [float(solve_structure(mm.eos_table(dict(P, L_sym=x)), hc, rtol=1e-11,
                                 atol=1e-13).Lambda) for x in Ls]
    slope = np.diff(lam) / np.diff(Ls)
    assert np.ptp(slope) < 0.02 * np.abs(slope).mean()
    rt = 1e-11
    J, _, _ = sensitivity_matrix(mm, P, ["L_sym"], masses_lambda=(1.4,),
                                 masses_radius=(1.4,), include_mmax=False,
                                 hc_grid_n=220, rtol=rt, atol=rt * 1e-2)

    def at14(q):
        """(Lambda, R) at M = 1.4 exactly (Newton on h_c)."""
        t = mm.eos_table(q)
        f = build_family(t, default_hc_grid(t, n_models=220), rtol=rt,
                         atol=rt * 1e-2)
        b = np.flatnonzero(np.asarray(f.stable))
        h = float(np.interp(1.4, np.asarray(f.M_msun)[b], np.asarray(f.h_c)[b]))
        for _ in range(3):
            s = lambda x: solve_structure(t, x, rtol=rt, atol=rt * 1e-2)
            h -= (float(s(h).M_msun) - 1.4) / float(jax.jacfwd(lambda x: s(x).M_msun)(h))
        pt = solve_structure(t, h, rtol=rt, atol=rt * 1e-2)
        return np.array([float(pt.Lambda), float(pt.R_km)])
    fd = (at14(dict(P, L_sym=L + 0.05)) - at14(dict(P, L_sym=L - 0.05))) / 0.1
    np.testing.assert_allclose(np.asarray(J)[:, 0], fd, rtol=0.01)


def test_invalid_core_rejected():
    """Metamodel, K_sym = 0: at low L the beta-equilibrium pressure stays on
    the 1e-9 floor past the join cap, so the core is rejected (error when
    eager, NaN table under jit); from L ~ 28 it joins."""
    import jax
    from nmbackend.eos.crust import InvalidCoreError
    mm = MetaModelEOS()
    for L in (20.0, 23.0, 25.0):
        with pytest.raises(InvalidCoreError):
            mm.eos_table({"L_sym": L, "K_sym": 0.0})
        t = jax.jit(mm.eos_table)({"L_sym": L, "K_sym": 0.0})
        assert not np.all(np.isfinite(np.asarray(t.p)))
    t = mm.eos_table({"L_sym": 30.0, "K_sym": 0.0})
    p = np.asarray(t.p)
    assert np.all(np.isfinite(p)) and np.all(np.diff(p) > 0)


def test_join_monotone_and_core_untouched():
    """p, eps, n increase; above the power-law segment the core nodes are the
    model's own; the crust n rescaling is small."""
    from nmbackend.eos.metamodel import DEFAULT_PARAMS
    mm = MetaModelEOS(muons=True)
    t = mm.eos_table()
    n, e, p = (np.asarray(a) for a in (t.n, t.eps_mevfm3, t.p_mevfm3))
    assert np.all(np.diff(n) > 0) and np.all(np.diff(e) > 0)
    assert np.all(np.diff(p) > 0)
    n0, e0, p0 = (np.asarray(a) for a in mm._low_density(
        dict(DEFAULT_PARAMS), n_min=join_density()))
    nj, ej, pj = (np.asarray(a) for a in attach_crust(n0, e0, p0))
    k = np.ones(n0.size, bool)
    idx = np.searchsorted(nj, n0[k])
    np.testing.assert_allclose(nj[idx], n0[k], rtol=1e-14)
    np.testing.assert_allclose(ej[idx], e0[k], rtol=1e-12)
    np.testing.assert_allclose(pj[idx], p0[k], rtol=1e-12)
    top = nj[np.searchsorted(nj, N_T_DEFAULT * 1.02) - 1]   # rescaled n_t
    assert abs(top / N_T_DEFAULT - 1.0) < 0.02
