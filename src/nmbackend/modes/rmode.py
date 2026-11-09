"""Numerical toroidal r-mode Galerkin benchmark, Newtonian O(Omega).

Projects i*omega_ro*u+2*Omega*z_cross_u+grad(delta_h)=0 onto the
l=m toroidal spherical harmonic. No GR, centrifugal distortion, radiation,
viscosity or finite-spin error estimate is supplied by this benchmark.
"""
from __future__ import annotations
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import lpmv
from .pencil import Pencil


def toroidal_rmode(m,omega,*,angular_points=24):
    """Assemble the Coriolis/mass pencil; do not insert the analytic frequency.

    For a barotropic star only l=m is exposed as this purely toroidal branch.
    Frequencies and Omega share angular-frequency units (not cyclic Hz).
    The radial integral for the leading r^m velocity profile cancels from the
    ratio; this does not recover EOS-dependent relativistic r modes.
    """
    if int(m)!=m or m<2 or angular_points<2*m+2 or not np.isfinite(omega):
        raise ValueError('Require integer m>=2, resolved angular grid and finite Omega')
    m=int(m)
    mu,weight=leggauss(angular_points)
    p=lpmv(m,m,mu)
    previous=lpmv(m,m-1,mu)
    derivative=(m*mu*p-(m+m)*previous)/(mu*mu-1)
    sine=np.sqrt(1-mu*mu)
    theta=-1j*m*p/sine
    phi=-sine*derivative
    mass=2*np.pi*np.sum(weight*(abs(theta)**2+abs(phi)**2))
    coriolis=2*np.pi*np.sum(weight*(theta.conj()*(-2*omega*mu*phi)
                                   +phi.conj()*(2*omega*mu*theta)))
    h=1j*coriolis
    # omega_ro = sigma_in + m*Omega in the paper's convention.
    k0=np.array([[h-m*omega*mass]],dtype=complex)
    return Pencil(k0,np.array([[-mass]]),np.zeros((1,1))),{
        'model':'Newtonian barotropic l=m toroidal r-mode, leading order in Omega',
        'm':m,'ell':m,'omega':float(omega),'angular_points':angular_points,
        'radiation':'not included','equilibrium':'spherical Newtonian limit',
        'angular_mass_integral':float(mass),'projected_coriolis':complex(coriolis)}
