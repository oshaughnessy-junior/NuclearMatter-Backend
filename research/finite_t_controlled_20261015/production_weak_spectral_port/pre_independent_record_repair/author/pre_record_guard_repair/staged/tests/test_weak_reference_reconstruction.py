"""Finite-range source/DOS bookkeeping, explicit omitted-range qualification."""
from dataclasses import replace
import math

import numpy as np
import pytest

from nmbackend.reference import weak_reference as W
from nmbackend.reference import _weak_spectral_common as S

BATH=W.FreeElectronBath(*S.FIXED_BATH)


def test_one_helicity_blackbody_DOS():
    from scipy.integrate import quad
    number=quad(lambda x:x*x/(1+math.exp(x)),0.,50.,epsabs=1e-12)[0]
    assert number==pytest.approx(1.8030853547393915,rel=3e-13)
    energy=quad(lambda x:x**3/(1+math.exp(x)),0.,50.,epsabs=1e-12)[0]
    assert energy==pytest.approx(7*math.pi**4/120,rel=3e-13)
    assert S.dos(3.)==3.**2/(2*math.pi**2*S.C.HC**3)


def test_immutable_energy_rule_guards():
    for args in [((1.,1.),(1.,1.),0.,2.),((1.,),(True,),0.,2.),((float('nan'),),(2.,),0.,2.),((1.,),(1.,),0.,2.),([1.],(2.,),0.,2.)]:
        with pytest.raises(ValueError):W.EnergyRule(*args)
    for args in [(0.,2.,True),(2.,1.,4),(0.,float('inf'),4)]:
        with pytest.raises(ValueError):W.EnergyRule.gauss(*args)


@pytest.mark.parametrize('channel,lo,hi',[('EC',2.,3.),('PC',2.,3.),('NEUTRON_DECAY',.3,.5)])
def test_actual_finite_range_empty_and_LTE(channel,lo,hi):
    rule=W.EnergyRule.gauss(lo,hi,2);controls=W.SpectralControls()
    rad=W.Radiation.fd(2.,BATH.equilibrium_mu_nue,BATH.equilibrium_mu_antinue)
    for radiation in (W.Radiation.empty(),rad):
        r=W.reconstruct_spectral_moments(channel,BATH,radiation,rule,controls)
        assert r.complete_finite_range and not r.whole_moment_qualified
        assert len(r.nodes)==2 and r.rule==rule and r.controls==controls
        fs=[0.,0.];bs=[0.,0.]
        for energy,weight,node in zip(rule.nodes,rule.weights,r.nodes):
            if radiation.mode=='empty':f=0.
            else:
                mu=radiation.mu_nue if channel=='EC'else radiation.mu_antinue
                f=1/(1+math.exp((energy-mu)/2.))
            for i in (0,1):
                fs[i]+=weight*S.dos(energy)*energy**i*node.j.value*(1-f)
                bs[i]+=weight*S.dos(energy)*energy**i*node.true_chi.value*f
        for i in (0,1):
            assert r.forward[i].value==pytest.approx(fs[i],rel=3e-14)
            assert r.backward[i].value==pytest.approx(bs[i],rel=3e-14)
            scale=fs[i]+bs[i]
            assert abs(r.net[i].value-(fs[i]-bs[i]))/scale<3e-9
            if radiation.mode=='FD':assert abs(r.net[i].value)/scale<2e-9
        if radiation.mode=='empty':assert r.entropy_production is None


def test_reconstruction_rejects_state_radiation_domain_before_calls():
    controls=W.SpectralControls();rule=W.EnergyRule.gauss(.2,.8,2)
    with pytest.raises(ValueError):W.reconstruct_spectral_moments('EC',BATH,W.Radiation.fd(1.9,0.,0.),rule,controls)
    for radiation in (W.Radiation.empty(),W.Radiation.fd(2.,0.,0.)):
        with pytest.raises(ValueError):W.reconstruct_spectral_moments('EC',object(),radiation,rule,controls)
    with pytest.raises(ValueError):W.reconstruct_spectral_moments('NEUTRON_DECAY',BATH,W.Radiation.empty(),W.EnergyRule.gauss(1.,2.,2),controls)
    with pytest.raises(ValueError):W.reconstruct_spectral_moments('EC',BATH,W.Radiation.empty(),rule,None)
    with pytest.raises(TimeoutError):W.reconstruct_spectral_moments('EC',BATH,W.Radiation.empty(),rule,replace(controls,budget_seconds=1e-12))


def test_tampered_node_contract_rejected():
    # Consumer never accepts external node files/records. Deliberately corrupt
    # its private evaluator in this local test to exercise admission anyway.
    controls=W.SpectralControls();rule=W.EnergyRule.gauss(2.,3.,2)
    good=W.spectral_coefficients('EC',BATH,rule.nodes[0],controls)
    def wrong(*args):return good
    with pytest.raises(ArithmeticError):S.reconstruct('PC',BATH,W.Radiation.empty(),rule,controls,wrong)
    with pytest.raises(ValueError):replace(good,j=S.empty_coefficient())
