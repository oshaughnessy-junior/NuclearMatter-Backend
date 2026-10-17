"""Nonrelativistic ideal FD nucleons with independent constant rest offsets.

Spin2, common kinetic mass, no antiparticles or interactions. MeV/fm units.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
import math
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import quad_vec
from scipy.optimize import brentq
from scipy.special import expit

ROOT=Path(__file__).resolve().parent.parent
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/'thermodynamics'),str(ROOT/'response_reference'),str(ROOT/'allowed_opacity'),str(ROOT/'weak_rates')]
import lepton_reference
from free_density_response import PairState
from allowed_capture import OpticalState
from static_capture import Q

HBARC=lepton_reference.HBARC
MASS=939.


@dataclass(frozen=True)
class Species:
    T:float
    mass:float
    offset:float
    kinetic_mu:float
    density:float
    kinetic_energy_density:float
    energy_density:float
    pressure:float
    entropy_density:float
    free_energy_density:float
    dn_dmu:float
    dn_dT_fixed_mu:float
    dmu_dT_fixed_density:float
    cv_fixed_density:float
    fluctuation_A0:float
    fluctuation_A1:float
    fluctuation_A2:float
    fermi_momentum:float
    temperature_over_mass:float
    fermi_momentum_over_mass:float

    @property
    def mu(self):return self.offset+self.kinetic_mu

    def asdict(self):return asdict(self)|{'mu':self.mu}


def state_mu(T,nu,offset=0.,mass=MASS,*,rtol=2e-12,tail=60.):
    if not all(math.isfinite(x) for x in (T,nu,offset,mass,rtol,tail)):
        raise ValueError('Finite arguments required')
    if T<=0 or mass<=0 or not 30<=tail<=100 or not 0<rtol<1:
        raise ValueError('Require T,mass>0; tail30..100;0<rtol<1')
    end=max(0.,nu)+tail*T
    breaks=sorted(set([0.,end]+[x for x in (nu-tail*T,nu-12*T,nu-4*T,nu,nu+4*T,nu+12*T) if 0<x<end]))
    def integrand(k):
        x=(k-nu)/T;f=expit(-x);w=f*expit(x)
        entropy=np.logaddexp(0.,-abs(x))+abs(x)*expit(-abs(x))
        return math.sqrt(k)*np.array([f,k*f,entropy,w,(k-nu)*w,(k-nu)**2*w])
    moments=np.zeros(6)
    for lo,hi in zip(breaks[:-1],breaks[1:]):
        part,error,info=quad_vec(integrand,lo,hi,epsabs=1e-24,epsrel=rtol,norm='max',limit=500,full_output=True)
        if not info.success and not (info.status==2 and error<=max(1e-24,rtol*np.max(abs(part)))):
            raise ArithmeticError(f'Quadrature: {info.message}')
        moments+=part
    prefactor=math.sqrt(2.)*mass**1.5/(math.pi**2*HBARC**3)
    n,K,s,A0,A1,A2=prefactor*moments
    if n<=0 or A0<=0:raise ArithmeticError('Density/susceptibility underflow: outside represented domain')
    variance=A2-A1*A1/A0
    if variance<0:raise ArithmeticError('Negative covariance heat capacity')
    pF=HBARC*(3*math.pi**2*n)**(1/3)
    return Species(float(T),float(mass),float(offset),float(nu),n,K,K+offset*n,2*K/3,s,
                   K+offset*n-T*s,A0/T,A1/T**2,-A1/(T*A0),variance/T**2,A0,A1,A2,
                   pF,T/mass,pF/mass)


def state_density(T,n,offset=0.,mass=MASS,**kwargs):
    if not all(math.isfinite(x) for x in (T,n,offset,mass)) or T<=0 or n<=0 or mass<=0:
        raise ValueError('Require finite T,n,mass>0 and finite offset; exact vacuum is not represented')
    wavelength=math.sqrt(2*math.pi*HBARC**2/(mass*T))
    eta_classical=math.log(n*wavelength**3/2)
    EF=HBARC**2*(3*math.pi**2*n)**(2/3)/(2*mass)
    lo=min(-2.,eta_classical-5.)
    hi=max(2.,EF/T+5.)
    eta=brentq(lambda x:state_mu(T,T*x,offset,mass,**kwargs).density/n-1,lo,hi,xtol=2e-13,rtol=2e-13)
    result=state_mu(T,T*eta,offset,mass,**kwargs)
    if abs(result.density/n-1)>1e-9:raise ArithmeticError('Density inversion failed residual gate')
    return result


def cold_state(n,offset=0.,mass=MASS):
    if not all(math.isfinite(x) for x in (n,offset,mass)) or n<=0 or mass<=0:raise ValueError('Positive finite n,mass required')
    pF=HBARC*(3*math.pi**2*n)**(1/3)
    EF=pF*pF/(2*mass)
    return {'density':n,'pF':pF,'EF':EF,'mu':offset+EF,'energy_density':offset*n+.6*n*EF,
            'kinetic_energy_density':.6*n*EF,'pressure':.4*n*EF,'entropy_density':0.}


@dataclass(frozen=True)
class Matter:
    neutron:Species
    proton:Species
    electron:object

    @property
    def density(self):return self.neutron.density+self.proton.density
    @property
    def Ye(self):return self.proton.density/self.density
    @property
    def baryon_energy_density(self):return self.neutron.energy_density+self.proton.energy_density
    @property
    def baryon_pressure(self):return self.neutron.pressure+self.proton.pressure
    @property
    def baryon_cv_fixed_densities(self):return self.neutron.cv_fixed_density+self.proton.cv_fixed_density

    def pair(self,initial='neutron'):
        if initial=='neutron':a,b=self.neutron,self.proton
        elif initial=='proton':a,b=self.proton,self.neutron
        else:raise ValueError('initial must be neutron or proton')
        return PairState(a.T,a.mass,a.offset,b.offset,a.mu,b.mu)

    def optical(self,flavor='nue'):
        if flavor=='nue':return OpticalState(self.pair('neutron'),self.electron.chemical_potential)
        if flavor=='antinue':return OpticalState(self.pair('proton'),-self.electron.chemical_potential)
        raise ValueError('flavor must be nue or antinue')


def matter_state(T,nn,np_,*,mass=MASS,offset_neutron=Q,offset_proton=0.):
    """Charge-neutral free n,p,e+- state; no photons added by this factory.

    Default offsets subtract one common nucleon rest energy while retaining Q.
    No weak beta equilibrium is imposed. nn,np_ must both be positive.
    """
    neutron=state_density(T,nn,offset_neutron,mass)
    proton=state_density(T,np_,offset_proton,mass)
    electron=lepton_reference.state_density(T,np_)
    return Matter(neutron,proton,electron)
