"""Standard Skyrme EDF: SLy4 reference checks and the saturation inversion."""
import math

import numpy as np
import jax.numpy as jnp
import pytest

from nmbackend.eos import SkyrmeEOS, SLY4, BSKG3
from nmbackend.eos.skyrme import (_eps_p, _solve_xp, effective_masses,
                                  saturation_properties,
                                  skyrme_from_saturation, skyrme_from_t)
from nmbackend.durca import durca_onset, durca_mass
from nmbackend.tov import build_family, default_hc_grid


def _star(params):
    m = SkyrmeEOS()
    tab, comp = m.eos_table(params), m.composition(params)
    fam = build_family(tab, default_hc_grid(tab, n_models=80))
    M = np.asarray(fam.M_msun)
    i = int(np.argmax(np.where(np.asarray(fam.stable), M, -np.inf)))
    n_c = float(np.exp(np.interp(float(fam.h_c[i]), np.asarray(tab.h),
                                 np.log(np.asarray(tab.n)))))
    on = durca_onset(comp)
    return M[i], n_c, on, durca_mass(tab, on.n_DU, fam.h_c[i]), comp


def test_sly4_saturation():
    """Fortin et al. 2016, arXiv:1604.01944, Table IV; m* from Chabanat 1998."""
    s = saturation_properties(SLY4)
    ref = dict(n0=0.159, BA=-15.97, K=230.0, E_sym=32.04, L=46.0,
               ms_frac=0.69, mv_frac=0.80)
    tol = dict(n0=0.001, BA=0.02, K=0.5, E_sym=0.06, L=0.2, ms_frac=0.01,
               mv_frac=0.01)
    for k, v in ref.items():
        assert abs(s[k] - v) < tol[k], (k, s[k])


def test_sly4_star_and_durca():
    """Douchin & Haensel, A&A 380, 151 (2001), astro-ph/0111092: M_max =
    2.05 Msun at n_c = 1.21 fm^-3, and the dUrca threshold lies above n_c,
    so no SLy star runs nucleonic dUrca.  Fortin et al. 2016 Table IV gives
    n_DU = 1.14 fm^-3, where x_p reaches 1/9."""
    mmax, n_c, on, mdu, comp = _star(SLY4)
    assert abs(mmax - 2.05) < 0.02 and abs(n_c - 1.21) < 0.03
    assert on.n_DU > n_c and math.isinf(mdu)
    n, xp = np.asarray(comp.n), np.asarray(comp.x_p)
    assert abs(np.interp(1.0 / 9.0, xp, n) - 1.14) < 0.02


def test_inversion_roundtrip_sly4():
    s = saturation_properties(SLY4)
    c = skyrme_from_saturation(n0=s["n0"], BA=s["BA"], K=s["K"],
                               ms_frac=s["ms_frac"], mv_frac=s["mv_frac"],
                               J=s["E_sym"], L=s["L"])
    for k in ("sigma", "a0", "b0", "a3", "b3", "c1", "c2"):
        assert abs(c[k] - SLY4[k]) < 1e-3 * max(1.0, abs(SLY4[k])), k


def test_thermodynamic_pressure():
    n = np.linspace(0.1, 1.2, 1101)
    nj = jnp.asarray(n)
    xp = _solve_xp(nj, SLY4)
    eps, pres = (np.asarray(a) for a in _eps_p(nj, xp, SLY4))
    np.testing.assert_allclose(pres[1:-1], (n**2 * np.gradient(eps / n, n))[1:-1],
                               rtol=2e-3, atol=2e-2)


def test_composition_charge_neutral():
    comp = SkyrmeEOS().composition(SLY4)
    x_p, x_e, x_mu = (np.asarray(a) for a in (comp.x_p, comp.x_e, comp.x_mu))
    np.testing.assert_allclose(x_e + x_mu, x_p, rtol=1e-8)


def test_sigma_from_K():
    """Stiffer K needs a larger density exponent sigma."""
    s = [skyrme_from_saturation(K=K)["sigma"] for K in (210.0, 230.0, 260.0)]
    assert s[0] < s[1] < s[2]
    with pytest.raises(ValueError):
        skyrme_from_saturation(K=100.0)


def test_bskg3_saturation():
    """Extended Skyrme: Grams et al. 2023, arXiv:2307.14276, Table 3."""
    s = saturation_properties(BSKG3)
    ref = dict(n0=0.15784, BA=-16.082, K=242.56, E_sym=31.000, L=55.061,
               K_sym=-21.248, ms_frac=0.8586, mv_frac=0.72171)
    tol = dict(n0=2e-4, BA=0.01, K=0.1, E_sym=0.01, L=0.02, K_sym=0.05,
               ms_frac=2e-4, mv_frac=2e-4)
    for k, v in ref.items():
        assert abs(s[k] - v) < tol[k], (k, s[k])


def test_bskg3_star_and_durca():
    """arXiv:2307.14276, Table 4: M_max = 2.26, M_dUrca = 1.38 Msun.  R_1.4
    (12.73 km there) is pinned in test_crust.py."""
    mmax, _, _, mdu, _ = _star(BSKG3)
    assert abs(mmax - 2.26) < 0.01 and abs(mdu - 1.38) < 0.01


def test_extended_inversion_roundtrip_bskg3():
    """Effective masses at n0, 2 n0, 4 n0 plus the saturation point recover
    all BSkG3 uniform-matter coefficients."""
    s = saturation_properties(BSKG3)
    ms, mv = zip(*(effective_masses(BSKG3, f * s["n0"]) for f in (1, 2, 4)))
    c = skyrme_from_saturation(n0=s["n0"], BA=s["BA"], K=s["K"],
                               ms_frac=ms[0], mv_frac=mv[0], J=s["E_sym"],
                               L=s["L"], ms_hi=ms[1:], mv_hi=mv[1:])
    for k in ("sigma", "a0", "b0", "a3", "b3", "c1", "c2", "c1b", "c2b",
              "c1g", "c2g"):
        assert abs(c[k] - BSKG3[k]) < 1e-4 * max(1.0, abs(BSKG3[k])), k


def test_extended_frees_ksym():
    """At fixed (n0, B/A, K, J, L, m*(n0)), the high-density effective
    masses move K_sym; the standard-Skyrme masses give standard Skyrme."""
    base = dict(K=240.0, ms_frac=0.8, mv_frac=0.75, J=31.0, L=50.0)
    cs = skyrme_from_saturation(**base)
    std = saturation_properties(cs)
    ms, mv = zip(*(effective_masses(cs, f * std["n0"]) for f in (2, 4)))
    flat = saturation_properties(skyrme_from_saturation(
        **base, ms_hi=ms, mv_hi=mv))
    low = saturation_properties(skyrme_from_saturation(
        **base, ms_hi=(0.65, 0.5), mv_hi=(0.75, 0.75)))
    assert abs(flat["K_sym"] - std["K_sym"]) < 0.1
    assert abs(low["L"] - 50.0) < 0.05 and abs(low["K_sym"] - std["K_sym"]) > 20.0
