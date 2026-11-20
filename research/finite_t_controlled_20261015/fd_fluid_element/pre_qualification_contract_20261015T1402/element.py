"""Transparent isochoric n,p,e+- fluid element with exact FD EOS closure.

Common nucleon rest energy is subtracted; neutron Q offset remains in u.
"""
from __future__ import annotations
from pathlib import Path
import sys,math,time
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'quantum_slab'))
import fixed_bath as physics
DOS=physics.DOS
EDGES=(0.,.1,.5,.78233341,1.80433131,4.,8.,16.,24.,40.,64.,100.)

def eos(T,nB,Ye,*,photons=False):
    if not all(math.isfinite(x) for x in (T,nB,Ye)) or T<=0 or nB<=0 or not 0<Ye<1:raise ValueError('Require finite positive T,nB and0<Ye<1')
    if not isinstance(photons,bool):raise ValueError('photons must be explicit bool')
    m=physics.matter_state(T,nB*(1-Ye),nB*Ye)
    n,p,e=m.neutron,m.proton,m.electron
    U=m.baryon_energy_density+e.energy_density
    Cv=m.baryon_cv_fixed_densities+e.cv_fixed_net_density
    A_n=n.mu-T*n.dmu_dT_fixed_density;A_p=p.mu-T*p.dmu_dT_fixed_density
    A_e=e.chemical_potential-T*e.dmu_dtemperature_fixed_net_density
    if photons:
        radiation=math.pi**2*T**4/(15*physics.HBARC**3)
        U+=radiation;Cv+=4*radiation/T
    if Cv<=0:raise ArithmeticError('Nonpositive FD heat capacity')
    return dict(matter=m,u_MeV_per_baryon=U/nB,u_T=Cv/nB,u_Y=A_p-A_n+A_e,photons=photons)


def source(T,nB,Ye,*,outer_order=8,internal_order=48,tail=40.,edges=EDGES):
    if isinstance(outer_order,bool) or not isinstance(outer_order,int) or outer_order<4:raise ValueError('outer_order integer>=4 required')
    physics.guarded_capture.validate_controls(1.,internal_order,internal_order,tail)
    es=np.asarray(edges,float)
    if es.ndim!=1 or len(es)<2 or es[0]!=0 or not np.all(np.isfinite(es)) or np.any(np.diff(es)<=0):raise ValueError('Invalid energy edges')
    state=eos(T,nB,Ye);m=state['matter'];xs,ws=leggauss(outer_order);moments={};panels=[];start=time.monotonic()
    for flavor in ('nue','antinue'):
        total=np.zeros(2)
        for lo,hi in zip(es[:-1],es[1:]):
            value=np.zeros(2)
            for x,w in zip(xs,ws):
                E=float(lo+(x+1)*(hi-lo)/2)
                c=physics.coefficients(m,flavor,E,order=internal_order,tail=tail)
                N=DOS*E*E*c['j_s'];value+=float(w*(hi-lo)/2)*np.array([N,E*N])
            total+=value;panels.append(dict(flavor=flavor,lo=float(lo),hi=float(hi),values=value.tolist()))
        moments[flavor]=dict(number_fm3_s=float(total[0]),energy_MeV_fm3_s=float(total[1]))
    F=(moments['antinue']['number_fm3_s']-moments['nue']['number_fm3_s'])/nB
    q=(moments['nue']['energy_MeV_fm3_s']+moments['antinue']['energy_MeV_fm3_s'])/nB
    return dict(T=T,nB=nB,Ye=Ye,outer_order=outer_order,internal_order=internal_order,tail=tail,edges=es.tolist(),moments=moments,panels=panels,dYe_dt=F,energy_loss_MeV_per_baryon_s=q,dT_dt_no_photons=(-q-state['u_Y']*F)/state['u_T'],runtime_seconds=time.monotonic()-start)


def recover_temperature(u,nB,Ye,Tguess,*,photons=False):
    if not all(math.isfinite(x) for x in (u,Tguess)) or Tguess<=0:raise ValueError('Finite target energy and positive guess required')
    fn=lambda T:eos(T,nB,Ye,photons=photons)['u_MeV_per_baryon']-u
    lo=Tguess/2;hi=Tguess*2
    for _ in range(20):
        if fn(lo)<=0<=fn(hi):break
        if fn(lo)>0:lo/=2
        if fn(hi)<0:hi*=2
    else:raise ArithmeticError('Energy inversion did not bracket positive T')
    return float(brentq(fn,lo,hi,xtol=1e-12,rtol=1e-13))
