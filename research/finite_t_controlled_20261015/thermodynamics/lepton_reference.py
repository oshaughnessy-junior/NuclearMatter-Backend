"""Independent-composition relativistic ideal charged-lepton reference.

Units: T, mu, mass in MeV (mu includes rest mass), densities in fm^-3,
energy/pressure/free energy in MeV fm^-3; entropy and cv in fm^-3 (k_B=1).
NumPy/SciPy only; deterministic CPU reference, not a JAX production kernel.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import numpy as np
from scipy.integrate import quad_vec
from scipy.optimize import brentq
from scipy.special import expit

HBARC = 197.3269804  # MeV fm; named convention, not a precision assertion
ELECTRON_MASS = 0.51099895069  # MeV
MUON_MASS = 105.6583755  # MeV
C = 1.0 / (math.pi**2 * HBARC**3)  # spin degeneracy 2


@dataclass(frozen=True)
class State:
    temperature: float
    chemical_potential: float
    mass: float
    net_density: float
    particle_density: float
    antiparticle_density: float
    energy_density: float
    pressure: float
    entropy_density: float
    free_energy_density: float
    cv_fixed_net_density: float
    dn_dmu_fixed_temperature: float
    dn_dtemperature_fixed_mu: float
    dmu_dtemperature_fixed_net_density: float

    def asdict(self):
        return asdict(self)


def state_mu(temperature, chemical_potential, mass=ELECTRON_MASS, *, rtol=2e-12, tail=60.0):
    """Evaluate a pair-equilibrated gas with fixed net conserved charge.

    Photons, Coulomb/interaction corrections, magnetic fields and baryons
    are excluded. T > 0; m >= 0. Negative net density is supported by symmetry.
    """
    T, mu, m = map(float, (temperature, chemical_potential, mass))
    if not all(math.isfinite(x) for x in (T, mu, m)) or T <= 0 or m < 0:
        raise ValueError('Require finite T > 0, finite mu, m >= 0')
    if not 30 <= tail <= 100:
        raise ValueError('tail must lie in [30, 100]')
    sign = 1.0 if mu >= 0 else -1.0
    u = abs(mu)
    # Integrate kinetic energy k=E-m. Explicit Fermi-surface breaks keep
    # entropy/susceptibility contributions resolved even for mu/T >> 1.
    end = max(0.0, u - m) + tail*T
    breaks = sorted(set([0.0, end] + [k for k in
        (u-m-tail*T, u-m, u-m+tail*T) if 0 < k < end]))

    def integrand(k):
        E = m + k
        p = math.sqrt(k*(k+2*m))
        jac = p*E  # p^2 dp = p E dk
        x = (E-u)/T
        y = (E+u)/T
        a, b = expit(-x), expit(-y)
        # Stable subtraction at mu~0, without sinh overflow at mu/T>>1.
        diff = a * (-math.expm1(-2*u/T)) / (1 + math.exp(-y))
        wa, wb = expit(x)*a, expit(y)*b
        def h(z):
            z = abs(z)
            return np.logaddexp(0.0, -z) + z*expit(-z)
        a0 = wa+wb
        a1 = (E-u)*wa-(E+u)*wb
        a2 = (E-u)**2*wa+(E+u)**2*wb
        return jac*np.array([diff, a, b, E*(a+b),
                            p*p/(3*E)*(a+b) if E else 0,
                            h(x)+h(y), a0, a1, a2])

    moments = np.zeros(9)
    for lo, hi in zip(breaks[:-1], breaks[1:]):
        part, error, info = quad_vec(integrand, lo, hi, epsabs=1e-18,
                               epsrel=rtol, norm='max', limit=500, full_output=True)
        if not info.success and not (info.status == 2 and
                error <= max(1e-18, rtol*np.max(np.abs(part)))):
            raise ArithmeticError(f'Quadrature failed: {info.message}; esterr={error}; norm={np.max(np.abs(part))}; interval={lo,hi}')
        if not np.all(np.isfinite(part)):
            raise ArithmeticError('Non-finite quadrature')
        moments += part
    n, npart, nanti, energy, pressure, entropy, a0, a1, a2 = C*moments
    if a0 <= 0:
        raise ArithmeticError('Susceptibility underflow; state outside reference domain')
    cv = (a2 - a1*a1/a0)/(T*T)
    if cv < -1e-12*max(a2/(T*T), 1e-300):
        raise ArithmeticError('Negative fixed-density heat capacity')
    return State(T, mu, m, sign*n,
                 npart if sign > 0 else nanti,
                 nanti if sign > 0 else npart,
                 energy, pressure, entropy, energy-T*entropy, max(0., cv),
                 a0/T, sign*a1/(T*T), -sign*a1/(T*a0))


def state_density(temperature, net_density, mass=ELECTRON_MASS, **kwargs):
    """Invert monotone net charge density; retain net-zero pair plasma."""
    T, n, m = map(float, (temperature, net_density, mass))
    if not math.isfinite(n):
        raise ValueError('Require finite density')
    if n == 0:
        return state_mu(T, 0., m, **kwargs)
    target = abs(n)
    pf = HBARC*(3*math.pi**2*target)**(1/3)
    upper = max(math.hypot(pf, m), T, m)
    # Pair-symmetric states can have arbitrarily small mu at finite T.
    # A fixed absolute tolerance in MeV silently loses their net charge.
    try:
        pair = state_mu(T, 0., m, **kwargs)
    except ArithmeticError as exc:
        if 'Susceptibility underflow' not in str(exc):
            raise
    else:
        estimate = target/pair.dn_dmu_fixed_temperature
        if 0 < estimate < min(T, upper/2):
            upper = 2*estimate
    for _ in range(64):
        if state_mu(T, upper, m, **kwargs).net_density >= target:
            break
        upper *= 2
    else:
        raise ArithmeticError('Density inversion failed to bracket')
    fraction = brentq(lambda x: -1. if x == 0 else state_mu(T, upper*x, m, **kwargs).net_density/target-1,
                      0., 1., xtol=1e-13, rtol=2e-13)
    result = state_mu(T, math.copysign(upper*fraction, n), m, **kwargs)
    if abs(result.net_density/n-1) > 1e-9:
        raise ArithmeticError('Density inversion failed its relative charge residual gate')
    return result


def thermal_energy_density(state, *, rtol=2e-12, tail=60.):
    """E(T,n_net)-E(0,n_net), avoiding subtraction of large cold energies.

    Uses the fixed-net-density constraint to remove mu_0 times the vanishing
    number difference. This is a lepton-sector increment, not total matter heat.
    """
    T, u, m = state.temperature, abs(state.chemical_potential), state.mass
    pf = HBARC*(3*math.pi**2*abs(state.net_density))**(1/3)
    mu0 = math.hypot(pf, m)
    end = max(0., u-m, mu0-m)+tail*T
    breaks = sorted(set([0.,end] + [k for k in
        (u-m-tail*T,u-m,u-m+tail*T,mu0-m) if 0 < k < end]))
    def integrand(k):
        E=m+k; p=math.sqrt(k*(k+2*m)); x=(E-u)/T
        diff = -expit(x) if E < mu0 else expit(-x)
        return p*E*((E-mu0)*diff+(E+mu0)*expit(-(E+u)/T))
    value=0.
    for lo,hi in zip(breaks[:-1],breaks[1:]):
        part,error,info=quad_vec(integrand,lo,hi,epsabs=1e-20,epsrel=rtol,limit=500,full_output=True)
        if not info.success and not (info.status == 2 and
                error <= max(1e-20, rtol*abs(part))):
            raise ArithmeticError(f'Thermal quadrature failed: {info.message}')
        value+=float(part)
    return C*value
