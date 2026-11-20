"""Selected INTERNAL factory/response regressions; independent anchors in study reports."""
from dataclasses import replace
import json
from pathlib import Path
import pytest
from nmbackend.reference import free_matter as F

DATA = json.loads((Path(__file__).parent/'data/free_matter_reference_internal.json').read_text())


def test_internal_forward_and_pair_convention():
    s = F.state_density(2., 1e-7, .3, photons='absent')
    assert s.muons_absent and s.model == F.MODEL
    assert (s.neutron.mass, s.proton.mass, s.electron.mass) == (F.MN, F.MP, F.ME)
    assert s.electron.net_density == pytest.approx(s.proton.particle_density, rel=1e-9)
    assert s.electron.particle_density > 5*s.electron.net_density
    assert s.electron.antiparticle_density > 5*s.electron.net_density
    for name in ('energy_density', 'pressure', 'cv_fixed_nB_Ye', 'denergy_dYe_fixed_nB_T', 'denergy_dnB_fixed_T_Ye'):
        assert getattr(s, name) == pytest.approx(DATA['initial_thermo'][name], rel=2e-9)
    assert all(b.log_fraction_upper_bound < -18.42 for b in s.anti_bounds)


def test_photons_are_explicit_and_no_composition_response():
    a = F.state_density(2., 1e-7, .3, photons='absent')
    b = F.state_density(2., 1e-7, .3, photons='trapped_LTE')
    assert b.energy_density-a.energy_density == pytest.approx(b.photon_energy_density)
    assert b.cv_fixed_nB_Ye-a.cv_fixed_nB_Ye == pytest.approx(b.photon_cv)
    assert (a.h_n, a.h_p, a.h_e) == (b.h_n, b.h_p, b.h_e)
    assert a.identity != b.identity


@pytest.mark.parametrize('args,kw', [
    ((True,1e-7,.3),{}), ((float('nan'),1e-7,.3),{}),
    ((.99,1e-7,.3),{}), ((2.,1e-9,.3),{}), ((2.,1e-7,0.),{}),
    ((2.,1e-7,1.),{}), ((2.,1e-7,.3),{'photons':'implicit'}),
    ((1.,2e-6,.89),{}),  # rectangle is not a guarantee of actual weak-bath admission
])
def test_admission(args,kw):
    with pytest.raises(ValueError):
        F.state_density(*args, **dict({'photons':'absent'}, **kw))


def test_forged_response_and_anti_bound_reject():
    s = F.state_density(2.,1e-7,.3,photons='absent')
    with pytest.raises(ValueError):
        F._canonical(replace(s, cv_fixed_nB_Ye=2*s.cv_fixed_nB_Ye))
    with pytest.raises(ValueError):
        F._canonical(replace(s, muons_absent=False))
    with pytest.raises(ValueError):
        F._anti('hypothetical omitted anti', replace(s.neutron, chemical_potential=-F.MN))
