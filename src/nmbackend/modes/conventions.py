"""Gao-Shibata frequency, frame and outgoing-wave conventions (Eqs. 2–3)."""
from __future__ import annotations
import numpy as np

HARMONIC_CONVENTION='exp(i*(sigma*t + m*phi))'


def frequencies(sigma,m,omega):
    """Signed inertial/corotating frequencies; angular speed uses the same units."""
    if not np.isfinite(sigma) or not np.isfinite(omega) or int(m)!=m:
        raise ValueError('Require finite frequencies and integer m')
    sigma=complex(sigma)
    return {'sigma':sigma,'omega_in':sigma.real,'omega_ro':sigma.real+m*omega,
            'rate':sigma.imag,'behavior':'damped' if sigma.imag>0 else 'growing' if sigma.imag<0 else 'conservative',
            'efolding_time':1/abs(sigma.imag) if sigma.imag else np.inf}


def conjugate_partner(sigma,m):
    return -np.conjugate(sigma),-m


def stationary_patterns(sigma,m,omega,*,angular_tolerance):
    """Neutral-pattern diagnostics, not proof of canonical-energy sign or growth."""
    if not np.isfinite(angular_tolerance) or angular_tolerance<0:
        raise ValueError('Require a nonnegative angular-frequency tolerance')
    info=frequencies(sigma,m,omega)
    return {'inertial_stationary':abs(info['omega_in'])<=angular_tolerance,
            'corotating_stationary':abs(info['omega_ro'])<=angular_tolerance}


def exterior_contour(s,rb,angle,*,stellar_outer_radius,sigma):
    """Validate outgoing decay and return complex r and dr/ds.

    This specifies a contour only. A real-grid RNS interpolant is not an
    analytic exterior metric or an assembled outgoing boundary condition.
    """
    s=np.asarray(s,dtype=float)
    if (not np.all(np.isfinite(s)) or np.any(s<0) or not np.isfinite(rb)
            or not np.isfinite(stellar_outer_radius) or stellar_outer_radius<=0
            or rb<=stellar_outer_radius or not np.isfinite(angle) or not np.isfinite(sigma)):
        raise ValueError('Invalid exterior contour/domain')
    jacobian=np.exp(1j*angle)
    if np.imag(sigma*jacobian)>=0:
        raise ValueError('Outgoing exp(-i*sigma*r) does not decay on this contour')
    return rb+s*jacobian,jacobian
