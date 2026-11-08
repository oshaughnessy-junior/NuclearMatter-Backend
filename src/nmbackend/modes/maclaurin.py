"""Finite-rotation Newtonian homogeneous Maclaurin m=2 bar-mode benchmark.

Ellipsoidal-potential integrals assemble the classical virial pencil; see
Baiotti et al. (astro-ph/0609473), Eqs. 7.1–7.3, and Braviner & Ogilvie
(1404.2461), Sect. 5. No relativistic or radiative-rate fit is used.
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from .pencil import Pencil


def bar_mode(eccentricity,*,dynamical_frequency=1.):
    """Assemble the inertial-frame polynomial in the exp(+i*sigma*t) convention.

    dynamical_frequency=sqrt(pi*G*density). Length scale a_equator=1 cancels
    from the frequencies and beta. Incompressibility is a Newtonian benchmark
    assumption, not a causal dense-matter EOS.
    """
    e=float(eccentricity)
    scale=float(dynamical_frequency)
    if not np.isfinite(e) or not 0<=e<1 or not np.isfinite(scale) or scale<=0:
        raise ValueError('Require 0<=eccentricity<1 and positive frequency scale')
    q=np.sqrt(1-e*e)
    def integrate(f):return quad(f,0.,np.inf,epsabs=2e-12,epsrel=2e-12,limit=200)
    a1,err1=integrate(lambda u:q/((1+u)**2*np.sqrt(q*q+u)))
    a3,err3=integrate(lambda u:q/((1+u)*(q*q+u)**1.5))
    b11,errb=integrate(lambda u:q*u/((1+u)**3*np.sqrt(q*q+u)))
    binding,errw=integrate(lambda u:1/((1+u)*np.sqrt(q*q+u)))
    omega_sq=2*(a1-q*q*a3) if e else 0.
    if omega_sq<0:raise ValueError('Unresolved equilibrium rotation coefficient')
    omega=np.sqrt(omega_sq)*scale
    b=b11*scale*scale
    # Standard opposite-time roots Omega +/- sqrt(4B-Omega^2) become
    # -Omega +/- sqrt(4B-Omega^2) here. Both signs are kept.
    pencil=Pencil([[2*omega*omega-4*b]],[[2*omega]],[[1.]])
    return pencil,{'model':'Newtonian homogeneous Maclaurin m=2 bar modes',
        'eccentricity':e,'axis_ratio':q,'m':2,'omega':float(omega),
        'beta':float(omega_sq/(2*q*binding)), 'b11':float(b),
        'discriminant':float(4*b-omega*omega),'dynamical_frequency':scale,
        'coefficient_quadrature_error':float(max(err1,err3,errb,errw)),
        'radiation':'not included','viscosity':'not included'}


def neutrality_thresholds():
    """Locate classical neutral-pattern and conservative dynamical thresholds."""
    def secular(e):return float(bar_mode(e)[0].k0[0,0].real)
    def dynamic(e):return bar_mode(e)[1]['discriminant']
    e_sec=brentq(secular,.7,.9,xtol=1e-12)
    e_dyn=brentq(dynamic,.9,.98,xtol=1e-12)
    return {'secular_pattern_e':e_sec,'secular_pattern_beta':bar_mode(e_sec)[1]['beta'],
            'dynamical_e':e_dyn,'dynamical_beta':bar_mode(e_dyn)[1]['beta']}
