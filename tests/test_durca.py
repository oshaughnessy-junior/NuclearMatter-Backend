"""beta-equilibrium composition and direct-Urca onset (nmbackend.durca)."""
import math

import numpy as np

from nmbackend.eos import RMFTEOS, MetaModelEOS, Composition
from nmbackend.eos.rmft import (GM1, couplings_from_saturation,
                                saturation_properties)
from nmbackend.durca import (NPE_THRESHOLD, durca_onset, durca_mass,
                             fermi_momenta)
from nmbackend.tov import build_family, default_hc_grid


def _npe(x_p_of_n, n):
    x_p = x_p_of_n(n)
    return Composition(n=n, x_p=x_p, x_e=x_p, x_mu=np.zeros_like(n))


def _m_du(model, params=None, n_models=60):
    tab = model.eos_table(params)
    comp = model.composition(params)
    fam = build_family(tab, default_hc_grid(tab, n_models=n_models))
    M = np.asarray(fam.M_msun)
    i = int(np.argmax(np.where(np.asarray(fam.stable), M, -np.inf)))
    on = durca_onset(comp)
    return on, durca_mass(tab, on.n_DU, fam.h_c[i])


def test_npe_triangle_closes_at_one_ninth():
    n = np.linspace(0.1, 1.0, 4001)
    comp = _npe(lambda n: 0.05 + 0.1 * n, n)
    on = durca_onset(comp)
    assert abs(on.x_p_DU - NPE_THRESHOLD) < 1e-6
    kn, kp, ke, _ = fermi_momenta(_npe(lambda n: NPE_THRESHOLD + 0 * n, n))
    np.testing.assert_allclose(kp + ke, kn, rtol=1e-12)
    assert math.isinf(on.n_DU_mu)          # no muons -> no muon channel


def test_no_onset_is_inf():
    n = np.linspace(0.1, 1.0, 200)
    on = durca_onset(_npe(lambda n: 0.05 + 0 * n, n))
    assert math.isinf(on.n_DU) and math.isnan(on.x_p_DU)
    assert durca_mass(None, on.n_DU) == math.inf


def test_rmft_composition_charge_neutral_with_muons():
    comp = RMFTEOS().composition()
    x_p, x_e, x_mu = (np.asarray(a) for a in (comp.x_p, comp.x_e, comp.x_mu))
    np.testing.assert_allclose(x_e + x_mu, x_p, rtol=1e-8)
    n = np.asarray(comp.n)
    assert x_mu[n < 0.1].max() == 0.0 and x_mu[n > 0.3].min() > 0.0


def test_metamodel_muons_option():
    comp = MetaModelEOS(muons=True).composition()
    x_p, x_e, x_mu = (np.asarray(a) for a in (comp.x_p, comp.x_e, comp.x_mu))
    np.testing.assert_allclose(x_e + x_mu, x_p, rtol=1e-8, atol=1e-12)
    n = np.asarray(comp.n)
    assert x_mu[np.argmin(abs(n - 0.32))] > 0.0
    # muons=False keeps the electron-only EOS (historical default)
    assert np.all(np.asarray(MetaModelEOS().composition().x_mu) == 0.0)


def test_muon_threshold_in_npemu_band():
    """With muons the electron-channel threshold x_p lies between the npe value
    1/9 and the ultra-relativistic limit (x_e = x_mu) 1/(1+(1+2^-1/3)^3)."""
    x_ur = 1.0 / (1.0 + (1.0 + 2.0 ** (-1.0 / 3.0)) ** 3)
    assert abs(x_ur - 0.1477) < 1e-4
    for comp in (RMFTEOS().composition(),
                 MetaModelEOS(muons=True).composition({"L_sym": 80.0,
                                                         "K_sym": 0.0})):
        on = durca_onset(comp)
        assert NPE_THRESHOLD < on.x_p_DU < x_ur
        assert on.n_DU_mu > on.n_DU          # muon channel opens later


def test_couplings_from_saturation_reproduce_gm1():
    c = couplings_from_saturation(n0=0.153, BA=-16.3, K=300.0, mstar_frac=0.70,
                                  J=32.5)
    for k in ("C_sigma", "C_omega", "C_rho", "b", "c"):
        assert abs(c[k] / GM1[k] - 1.0) < 0.01, k
    s = saturation_properties(couplings_from_saturation(L=60.0))
    assert abs(s["L"] - 60.0) < 1.0 and abs(s["E_sym"] - 32.5) < 0.2


def test_mdu_monotone_in_L_rmf_family():
    """Fixed GM1 saturation point and J; lowering L (omega-rho coupling) lowers
    x_p at high density, so M_DU rises and eventually leaves the stable branch."""
    rm = RMFTEOS()
    m = [_m_du(rm, couplings_from_saturation(L=L))[1] for L in (92, 85, 78, 60)]
    assert m[0] < m[1] < m[2] <= m[3]
    assert math.isinf(m[3])


def test_gm1_durca_onset_literature():
    """Raduta, Sedrakian & Weber, MNRAS 475, 4347 (2018), arXiv:1712.00584,
    Table 2 (GM1A: GM1 nucleonic couplings, purely nucleonic matter):
    n_DU = 0.28 fm^-3, M_DU = 1.10 Msun."""
    on, mdu = _m_du(RMFTEOS())
    assert abs(on.n_DU - 0.28) < 0.01
    assert abs(mdu - 1.10) < 0.05
