"""Cooling composition: Landau masses per family, D0 storage, dUrca agreement."""
import math

import jax
import numpy as np
import pytest

from nmbackend.eos import (DD2, DDME2, SLY4, DDRMFEOS, MetaModelEOS, RMFTEOS,
                           SkyrmeEOS)
from nmbackend.eos import ddrmf, metamodel, rmft, skyrme
from nmbackend.eos.coolcomp import (D0_GRID, cooling_composition, pack_d0,
                                    unpack_d0)
from nmbackend.eos.ddrmf import couplings_from_saturation_dd
from nmbackend.eos.rmft import couplings_from_saturation
from nmbackend.eos.skyrme import skyrme_from_saturation
from nmbackend.durca import durca_onset

jax.config.update("jax_enable_x64", True)


def _pop(c):
    c = dict(c)
    c.pop("L0")
    return c


# one representative draw per paper-6 family
FAMILIES = {
    "mm": (lambda: MetaModelEOS(muons=True), None),
    "dd": (DDRMFEOS, DD2),
    "ddf": (DDRMFEOS, _pop(couplings_from_saturation_dd(
        n0=0.152, BA=-16.0, K=250.0, mstar_frac=0.62, J=33.0, L=90.0,
        d_s=0.5, d_w=0.8, allow_negative_arho=True))),
    "rmf": (RMFTEOS, None),
    "rmfwr": (RMFTEOS, _pop(couplings_from_saturation(L=60.0))),
    "sk": (SkyrmeEOS, SLY4),
    "skx": (SkyrmeEOS, skyrme_from_saturation(ms_hi=(0.6, 0.5),
                                              mv_hi=(0.7, 0.6))),
}


def _dirac(mst, n_q, M_frac=1.0):
    """Dirac mass / M from the Landau mass E*_F / M at density n_q."""
    kf = (3.0 * math.pi**2 * n_q) ** (1.0 / 3.0)
    M = 939.0 / 197.3269804 * M_frac
    return math.sqrt((float(mst) * M) ** 2 - kf**2) / M


@pytest.mark.parametrize("fam", sorted(FAMILIES))
def test_composition_matches_durca_input(fam):
    """x_p is the composition durca.py uses, so the onset is identical; the
    D0 grid moves it by less than its resolution."""
    cls, par = FAMILIES[fam]
    m = cls()
    cc = cooling_composition(m, par)
    c0 = m.composition(par)
    np.testing.assert_array_equal(np.asarray(cc.x_p), np.asarray(c0.x_p))
    np.testing.assert_allclose(np.asarray(cc.x_n + cc.x_p), 1.0)
    on, on0 = durca_onset(cc), durca_onset(c0)
    assert on == on0 or (math.isinf(on.n_DU) and math.isinf(on0.n_DU))
    ond = durca_onset(unpack_d0(pack_d0(cc)))
    if on.n_DU <= D0_GRID[-1]:
        assert abs(ond.n_DU - on.n_DU) < 2e-3
    else:
        assert math.isinf(ond.n_DU)
    mn, mp = np.asarray(cc.mst_n), np.asarray(cc.mst_p)
    ok = np.asarray(cc.n) <= 1.0
    assert np.all((mn[ok] > 0.2) & (mn[ok] < 1.05) & (mp[ok] > 0.2)
                  & (mp[ok] < 1.05))


def test_sly4_published_masses():
    """Chabanat et al., NPA 635, 231 (1998): SLy4 m*/m = 0.695 and
    kappa_v = 0.25, i.e. m*_v/m = 0.80 (a proton in neutron matter)."""
    n0 = skyrme.saturation_properties(SLY4)["n0"]
    ms, _ = skyrme.landau_masses(SLY4, n0, 0.5)
    _, mv = skyrme.landau_masses(SLY4, n0, 0.0)
    assert abs(float(ms) - 0.695) < 0.003
    assert abs(float(mv) - 0.80) < 0.003


def test_extended_skyrme_masses_at_high_density():
    """skx draws fix m*_s, m*_v at 2 n0 and 4 n0; the export returns them."""
    p = FAMILIES["skx"][1]
    n0 = 0.160
    for f, ms, mv in ((2.0, 0.6, 0.7), (4.0, 0.5, 0.6)):
        s, _ = skyrme.landau_masses(p, f * n0, 0.5)
        _, v = skyrme.landau_masses(p, f * n0, 0.0)
        assert abs(float(s) - ms) < 1e-6 and abs(float(v) - mv) < 1e-6


@pytest.mark.parametrize("params, n0, mD", [
    # Typel et al. 2010 (arXiv:0908.2344), n0 and m_D/m
    (DD2, 0.149065, 0.5625),
    # Lalazissis et al., PRC 71, 024312 (2005)
    (DDME2, 0.152, 0.572),
])
def test_ddrmf_published_dirac_mass(params, n0, mD):
    """Landau mass E*_F in symmetric matter at n0 inverts to the published
    Dirac mass."""
    mn, mp = ddrmf.landau_masses(params, n0, 0.5)
    Mn = params["M_n"] / 939.0
    assert abs(_dirac(mn, n0 / 2, Mn) - mD) < 0.002
    assert float(mn) > mD and float(mp) > mD


@pytest.mark.parametrize("params", [None, _pop(couplings_from_saturation(
    L=60.0))])
def test_gm1_published_dirac_mass(params):
    """Glendenning & Moszkowski, PRL 67, 2414 (1991): GM1 m*/M = 0.70 at
    n0 = 0.153 fm^-3; the omega-rho coupling (rmfwr) leaves it unchanged."""
    mn, mp = rmft.landau_masses(params, 0.153, 0.5)
    assert abs(float(mn) - float(mp)) < 1e-12
    assert abs(_dirac(mn, 0.153 / 2) - 0.70) < 0.005


def test_metamodel_prescription():
    """Margueron+2018 Eqs. 15-16 with m*_sat/m = 0.75 in symmetric matter
    and m*_n - m*_p = 0.10 m in neutron matter, both at n_sat."""
    ns = metamodel.DEFAULT_PARAMS["n_sat"]
    mn, mp = metamodel.landau_masses(None, ns, 0.5)
    assert abs(float(mn) - 0.75) < 1e-12 and abs(float(mp) - 0.75) < 1e-12
    mn, mp = metamodel.landau_masses(None, ns, 0.0)
    assert abs(float(mn - mp) - 0.10) < 1e-12
    mn, mp = metamodel.landau_masses({"mstar_sat": 0.8, "dmstar_sat": 0.0},
                                     2 * ns, 0.1)
    assert abs(float(mn) - 1.0 / (1.0 + 2.0 * 0.25)) < 1e-12
    assert float(mp) == float(mn)


def test_jit_and_d0_shape():
    m = SkyrmeEOS()
    cc = jax.jit(lambda p: cooling_composition(m, p))(SLY4)
    a = pack_d0(cc)
    assert a.shape == (4, D0_GRID.size) and a.dtype == np.float32
    assert np.all(np.isfinite(a))
