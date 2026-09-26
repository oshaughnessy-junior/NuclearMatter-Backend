"""SLy crust join (nmbackend.eos.crust) and R_1.4 against published values."""
import functools

import jax
import numpy as np
import pytest

from nmbackend.eos import (DD2, SLY4, BSKG3, DDRMFEOS, MetaModelEOS, RMFTEOS,
                           SkyrmeEOS)
from nmbackend.eos.crust import N_T_DEFAULT, attach_crust
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


def test_sly4_matches_douchin_haensel():
    """Core and crust are both SLy4, so the unified value is the target:
    R_1.4 = 11.72 km (Douchin & Haensel 2001, astro-ph/0111092)."""
    assert abs(star14("sly4")[0] - 11.72) < 0.1


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
        dict(DEFAULT_PARAMS), n_min=N_T_DEFAULT * 1.001))
    nj, ej, pj = (np.asarray(a) for a in attach_crust(n0, e0, p0))
    k = n0 >= 1.5 * N_T_DEFAULT
    idx = np.searchsorted(nj, n0[k])
    np.testing.assert_allclose(nj[idx], n0[k], rtol=1e-14)
    np.testing.assert_allclose(ej[idx], e0[k], rtol=1e-12)
    np.testing.assert_allclose(pj[idx], p0[k], rtol=1e-12)
    top = nj[np.searchsorted(nj, N_T_DEFAULT * 1.02) - 1]   # rescaled n_t
    assert abs(top / N_T_DEFAULT - 1.0) < 0.02
