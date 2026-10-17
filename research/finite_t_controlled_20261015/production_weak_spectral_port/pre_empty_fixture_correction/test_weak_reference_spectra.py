"""Selected direct spectra/current/support gates, not a uniform-domain certificate."""
from dataclasses import replace, FrozenInstanceError
from pathlib import Path
import json
import math

import numpy as np
import pytest

from nmbackend.reference import weak_reference as W
from nmbackend.reference import _weak_spectral_capture as S
from nmbackend.reference import _weak_current as C
from nmbackend.reference import _weak_quadrature as Q

DATA=json.loads((Path(__file__).parent/'data/free_weak_spectral_reference.json').read_text())
BATH=W.FreeElectronBath(**DATA['spectra'][0]['bath'])


@pytest.mark.parametrize('row',DATA['spectra'])
def test_saved_direct_branches(row):
    r=W.spectral_coefficients(row['channel'],BATH,row['E_MeV'],W.SpectralControls())
    for key in ('j','true_chi'):
        v=getattr(r,key)
        assert v.support=='positive_support' and v.log is not None
        assert abs(math.expm1(v.log-row[key]['log']))<3e-6
    assert abs(r.kms_log_residual)<2e-9
    assert r.relaxation.value==pytest.approx(r.j.value+r.true_chi.value,rel=3e-15)
    assert r.absorption_cm_inverse.value==pytest.approx(r.true_chi.value/2.99792458e10,rel=3e-15)
    assert r.absorption_km_inverse.value==pytest.approx(r.true_chi.value/2.99792458e5,rel=3e-15)


def test_explicit_spectral_optin_preserves_stageA():
    with pytest.raises(NotImplementedError):W.spectral_coefficients('EC',BATH,2.)
    with pytest.raises(FrozenInstanceError):W.SpectralControls().tail=70.
    for sign in (-1,1):
        with pytest.raises(ValueError):W.SignedQuantity(0.,None,sign)
    assert Q.positive().reason=='absent_initial_neutrino_occupation'
    with pytest.raises(ValueError):W.SpectralQuantity(0.,None,'positive_support','absent_initial_neutrino_occupation')


@pytest.mark.parametrize('bad',[True,False,None,'2',float('nan'),float('inf'),-float('inf'),10**1000,-1.,0.,61.])
def test_spectral_inputs_before_physics(bad):
    with pytest.raises(ValueError):W.spectral_coefficients('EC',BATH,bad,W.SpectralControls())


def test_bath_domain_and_controls():
    with pytest.raises(ValueError):W.spectral_coefficients('EC',replace(BATH,mu_e=BATH.mu_e+1e-12),2.,W.SpectralControls())
    with pytest.raises(ValueError):W.spectral_coefficients('all',BATH,2.,W.SpectralControls())
    for kw in [dict(tail=True),dict(tail=29.),dict(outer_tail=61.),dict(phi_order=True),dict(energy_order=33),dict(budget_seconds=121.),dict(budget_seconds=float('nan'))]:
        with pytest.raises(ValueError):W.SpectralControls(**kw)
    with pytest.raises(ValueError):W.spectral_coefficients('NEUTRON_DECAY',BATH,1.41,W.SpectralControls())
    with pytest.raises(ValueError):W.capture_differential_coefficients('NEUTRON_DECAY',BATH,.5,2.,.1,W.SpectralControls())
    with pytest.raises(TimeoutError):W.spectral_coefficients('PC',BATH,2.,replace(W.SpectralControls(),budget_seconds=1e-12))
    with pytest.raises(TimeoutError):W.spectral_coefficients('NEUTRON_DECAY',BATH,.8,replace(W.SpectralControls(),budget_seconds=1e-12))


def test_support_linear_timelike_and_moving_threshold():
    with pytest.raises(ArithmeticError):S.support(C.MN,C.MP,1.,0.)
    # Exact linear support and the original unsquared positive-energy shell.
    for ma,mb,w,q in [(C.MN,C.MP,-1.,1.),(C.MP,C.MN,1.,1.),(C.MN,C.MP,-1.29,1.1)]:
        for lo,hi in S.support(ma,mb,w,q):
            ea=lo+1. if hi is None else (lo+hi)/2
            pa=math.sqrt((ea-ma)*(ea+ma));co=((ma-mb)*(ma+mb)+w*w-q*q+2*w*ea)/(2*pa*q)
            assert abs(co)<=1+1e-10 and ea+w>=mb
    r=W.capture_differential_coefficients('PC',BATH,1.7,.5987748936352157,-.1995027501115606,W.SpectralControls.fine())
    assert r.j.value==pytest.approx(.0032402781881332924,rel=3e-10)
    assert r.true_chi.value==pytest.approx(.0018607452915373856,rel=3e-10)
    e=W.capture_differential_coefficients('PC',BATH,5.,.6,1.,W.SpectralControls())
    assert e.j.value==e.true_chi.value==0 and e.j.support=='exact_kinematic_zero'
    with pytest.raises(ValueError):replace(e,differential_coordinates=None,units='s^-1 per occupation')
    with pytest.raises(ValueError):e.absorption_cm_inverse


def test_finite_log_not_falsezero():
    for log in (-14000.,1000.):
        reason='underflow' if log<0 else 'overflow'
        v=W.SpectralQuantity(None,log,'positive_support',reason)
        assert v.log==log
        with pytest.raises(ValueError):W.SpectralQuantity(0.,log,'positive_support',reason)
    for log in (None,True,float('nan'),float('inf'),-float('inf')):
        with pytest.raises(ValueError):W.SpectralQuantity(None,log,'positive_support','underflow')
    with pytest.raises(ValueError):W.SpectralQuantity(0.,None,'unresolved','zero')


def test_literal_current_on_fresh_delta_shell():
    for ch,E,el,c in [('EC',4.3,5.59,.27),('EC',4.3,5.59,.9999),('PC',1.73,.62,-.24)]:
        ma,mb=(C.MN,C.MP)if ch=='EC'else(C.MP,C.MN)
        pe=math.sqrt((el-C.ME)*(el+C.ME));w=E-el;q=math.sqrt((E-pe)**2+2*E*pe*(1-c))
        intervals=S.support(ma,mb,w,q);lo,hi=intervals[0]
        ea=lo+2. if hi is None else (lo+hi)/2
        p=math.sqrt((ea-ma)*(ea+ma));co=((ma-mb)*(ma+mb)+w*w-q*q+2*w*ea)/(2*p*q)
        assert abs(co)<=1
        pa=np.array([[ea,p*math.sqrt(1-co*co),0,p*co]]);pb=pa+np.array([w,0,0,q])
        kz=(E*E+q*q-pe*pe)/(2*q);kx=math.sqrt(max(0.,E*E-kz*kz))
        k=np.array([[E,kx,0,kz]]);l=np.array([[el,kx,0,kz-q]])
        pn,pp=(pa,pb)if ch=='EC'else(pb,pa)
        scalar=C.scalar_capture(ch,k,pn,l,pp)[0];literal=C.literal_capture(ch,k,pn,l,pp)[0]
        assert abs(scalar/literal-1)<3e-10
        assert abs(C.dot(pb,pb)[0]/mb**2-1)<1e-10
