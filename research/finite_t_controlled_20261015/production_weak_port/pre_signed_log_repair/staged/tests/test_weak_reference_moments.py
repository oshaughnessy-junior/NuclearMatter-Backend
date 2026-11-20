"""Selected physical moments, forward regressions and strict contract failures."""
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import json
import math

import numpy as np
import pytest

from nmbackend.reference import weak_reference as W
from nmbackend.reference import _weak_quadrature as Q
from nmbackend.reference import _weak_decay as D
from nmbackend.reference import _weak_current as C

FIXTURE = json.loads((Path(__file__).parent/'data/free_weak_reference.json').read_text())


@pytest.mark.parametrize('channel', ['EC', 'PC', 'NEUTRON_DECAY'])
def test_saved_absolute_moments_empty_LTE_noneq(channel):
    bath = W.FreeElectronBath(**FIXTURE['bath'])
    mu, anti = bath.equilibrium_mu_nue, bath.equilibrium_mu_antinue
    scenarios = [('empty', W.Radiation.empty()), ('LTE', W.Radiation.fd(bath.T, mu, anti))]
    if channel == 'NEUTRON_DECAY':
        scenarios += [('emission_dominated', W.Radiation.fd(bath.T, mu, anti-1.6)),
                      ('inverse_dominated', W.Radiation.fd(bath.T, mu, anti+1.6))]
    else:
        scenarios += [('noneq', W.Radiation.fd(bath.T, mu-.6*bath.T, anti-.8))]
    for name, radiation in scenarios:
        row = W.channel_moments(channel, bath, radiation)
        target = FIXTURE['channels'][channel]['saved'][name]
        for branch, label in [(row.forward, 'F'), (row.backward, 'B')]:
            for i in (0, 1):
                if target[label][i]:
                    assert abs(branch[i].value/target[label][i]-1) < 3e-6
                else:
                    assert branch[i].value == 0 and branch[i].log is None
        assert abs(row.energy_ledger_scaled) < 3e-9
        assert abs(row.direct_energy_ledger_scaled) < 3e-9
        assert row.max_event_kms_error < 2e-9
        if name == 'LTE':
            assert abs(row.net[0].value)/(row.forward[0].value+row.backward[0].value) < 2e-10
        if radiation.mode == 'FD':
            assert row.entropy_production.value >= 0
            mu_rad = radiation.mu_nue if channel == 'EC' else radiation.mu_antinue
            mu_eq = mu if channel == 'EC' else anti
            expected = row.net[0].value*(mu_eq-mu_rad)/bath.T
            scale = row.forward[0].value+row.backward[0].value
            assert abs(row.entropy_production.value-expected)/scale < 3e-9
        n = dict(row.particle_stoichiometry)
        assert n['neutron']+n['proton'] == 0
        assert n['proton']-n['electron_net'] == 0
        assert n['electron_net']+n.get('nue', 0)-n.get('antinue', 0) == 0


@pytest.mark.parametrize('bad', [True, False, float('nan'), float('inf'), -float('inf'), 10**1000, '2', None])
def test_nonfinite_nonbool_bath(bad):
    with pytest.raises(ValueError):
        W.FreeElectronBath(bad, 920., 918., .2)
    with pytest.raises(ValueError):
        W.FreeElectronBath(2., bad, 918., .2)
    with pytest.raises(ValueError):
        W.FreeElectronBath(2., 920., 918., bad)


@pytest.mark.parametrize('kwargs', [dict(T=.9), dict(T=3.1), dict(mu_n=C.MN), dict(mu_p=C.MP-500), dict(mu_e=7.)])
def test_bath_domain(kwargs):
    values = dict(FIXTURE['bath']);values.update(kwargs)
    with pytest.raises(ValueError):W.FreeElectronBath(**values)


@pytest.mark.parametrize('kwargs', [dict(radial_order=True), dict(polar_order=3),dict(rest_order=65),dict(tail=np.nan),dict(tail=29),dict(budget_seconds=True),dict(budget_seconds=121),dict(decay_coordinate='fake'),dict(parent_coordinate='fake'),dict(lepton_order=64,polar_order=64,azimuth_order=64)])
def test_control_rejection(kwargs):
    with pytest.raises(ValueError):W.Controls(**kwargs)


def test_radiation_and_scope_rejection():
    b = W.FreeElectronBath(**FIXTURE['bath'])
    for values in [('FD', -2., 0., 0.),('FD',2., True,0.),('FD',2.,0.,float('nan')),('FD',2.,21.,0.),('empty',2.,None,None),('empty',None,0.,None)]:
        with pytest.raises(ValueError):W.Radiation(*values)
    with pytest.raises(ValueError):W.channel_moments('EC', b, W.Radiation.fd(1.9,0.,0.))
    with pytest.raises(ValueError):W.channel_moments('all_urca', b)
    with pytest.raises(NotImplementedError):W.spectral_coefficients('EC',b,5.)
    with pytest.raises(FrozenInstanceError):b.T=2.1
    with pytest.raises(FrozenInstanceError):W.Current().weak_mass=12.


def test_positive_log_and_signed_representation():
    under = Q.positive(-1000.)
    assert under.value is None and under.log == -1000. and under.reason == 'underflow'
    over = Q.positive(1000.)
    assert over.value is None and over.reason == 'overflow'
    for value, log, reason in [(0.,-1000.,'underflow'),(None,None,'empty_support'),(0.,None,'empty_support'),(True,0.,'represented'),(1.,False,'represented'),(1.,float('nan'),'represented')]:
        with pytest.raises(ValueError):W.PositiveQuantity(value,log,reason)
    with pytest.raises(ValueError):W.SignedQuantity(True,0.,1)
    assert Q.signed(-1000.,-1).value is None


def test_timeout_never_returns_incomplete_or_zero():
    b=W.FreeElectronBath(**FIXTURE['bath'])
    with pytest.raises(TimeoutError):
        W.channel_moments('EC',b,controls=replace(W.Controls.base('EC'),budget_seconds=1e-12))
    with pytest.raises(TimeoutError):
        W.channel_moments('NEUTRON_DECAY',b,controls=replace(W.Controls.base('NEUTRON_DECAY'),budget_seconds=1e-12))


def test_two_rest_phase_coordinates_and_time_dilation():
    pair=D.rest_events(20,'pair');electron=D.rest_events(28,'electron')
    rate=sum(pair['weight']);reference=sum(electron['weight'])
    assert abs(rate/reference-1)<3e-6
    for index in [0,1,7]:
        p=pair['neutron'][index:index+1];pp=pair['proton'][index:index+1]
        el=pair['lepton'][index:index+1];nu=pair['neutrino'][index:index+1]
        beta=np.array([[0.,0.,.137]])
        lab=[C.boost(v,beta)for v in(p,pp,el,nu)]
        assert np.max(abs(lab[0]-lab[1]-lab[2]-lab[3]))<2e-10*C.MN
        gamma=1/math.sqrt(1-.137**2)
        assert lab[0][0,0]/C.MN==pytest.approx(gamma,rel=2e-15)
        assert abs(C.decay_amplitude(*lab)[0]/C.decay_amplitude(p,pp,el,nu)[0]-1)<3e-10
