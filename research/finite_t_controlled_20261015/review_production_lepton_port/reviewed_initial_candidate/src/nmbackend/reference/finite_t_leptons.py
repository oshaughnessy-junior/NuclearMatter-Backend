"""CPU ideal relativistic lepton pair states at independent net number or mu.

T, total chemical potential and mass are in MeV; number densities, entropy
(k_B=1) and fixed-net-number heat capacity are in fm^-3. Energy, pressure and
free-energy density are rest-inclusive MeV fm^-3. This scalar NumPy/SciPy
reference does not provide JAX differentiation or a complete matter EOS.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import numbers

import numpy as np
from scipy.integrate import quad_vec
from scipy.optimize import brentq
from scipy.special import expit

from ..constants import HBARC, M_E, M_MU

__all__ = ["LeptonState", "state_mu", "state_density", "thermal_energy_density"]
_MEASURE = 1.0 / (math.pi**2 * HBARC**3)  # two spin states


@dataclass(frozen=True)
class LeptonState:
    """One pair-equilibrated flavor; antiparticle mu is minus particle mu.

    net_density counts negatively charged leptons minus positive antiparticles;
    electric charge density is -e * net_density. No inter-flavor equilibrium or
    baryonic neutrality is imposed. Response fields hold the stated net number
    or chemical potential fixed, not the particle-only number. Susceptibility
    and dn/dT are in fm^-3 MeV^-1; dmu/dT is dimensionless.
    """
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
        """Return named scalar fields; units are those of this module."""
        return asdict(self)


def _real(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(name + " must be a finite real scalar, not Boolean")
    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(name + " is not represented as a finite float64") from exc
    if not math.isfinite(value):
        raise ValueError(name + " must be finite")
    return value


def _inputs(temperature, chemical_potential, mass, rtol, tail):
    T = _real("temperature_mev", temperature)
    mu = _real("chemical_potential_mev", chemical_potential)
    m = _real("mass_mev", mass)
    rtol = _real("rtol", rtol)
    tail = _real("tail", tail)
    if T <= 0 or m < 0 or not 0 < rtol < 1e-4 or not 30 <= tail <= 100:
        raise ValueError("Require T>0, mass>=0, 0<rtol<1e-4, 30<=tail<=100")
    return T, mu, m, rtol, tail


def _integrate(fun, breaks, rtol, epsabs):
    value = None
    for lo, hi in zip(breaks[:-1], breaks[1:]):
        part, error, info = quad_vec(fun, lo, hi, epsabs=epsabs,
                                     epsrel=rtol, norm="max", limit=500,
                                     full_output=True)
        magnitude = float(np.max(np.abs(part)))
        if (not np.all(np.isfinite(part)) or not math.isfinite(float(error))
                or (not info.success and not
                    (info.status == 2 and error <= max(epsabs, rtol*magnitude)))):
            raise ArithmeticError("Unresolved finite-temperature lepton quadrature")
        value = part if value is None else value + part
    if value is None or not np.all(np.isfinite(value)):
        raise ArithmeticError("Unrepresented finite-temperature lepton integral")
    return value


def state_mu(temperature_mev, chemical_potential_mev, mass_mev=M_E, *,
             rtol=2e-12, tail=60.0):
    """Evaluate one ideal pair gas at T>0 and total mu (both MeV).

    Signed mu is allowed; mu=0 retains a net-zero pair plasma. Photons,
    interactions and weak equilibrium are absent. Ordinary populations may
    underflow; no relative accuracy or exact absence is implied by such a zero.
    Unresolved susceptibility or nonfinite responses raise ArithmeticError.
    """
    T, mu, m, rtol, tail = _inputs(temperature_mev, chemical_potential_mev,
                                 mass_mev, rtol, tail)
    sign = 1.0 if mu >= 0 else -1.0
    u = abs(mu)
    end = max(0.0, u-m) + tail*T
    if not math.isfinite(end):
        raise ArithmeticError("Unrepresented finite-temperature energy interval")
    breaks = sorted(set([0.0, end] + [k for k in
                        (u-m-tail*T, u-m, u-m+tail*T) if 0 < k < end]))

    def integrand(k):
        E = m+k
        p = math.sqrt(k*(k+2*m))
        x, y = (E-u)/T, (E+u)/T
        a, b = expit(-x), expit(-y)
        diff = a * (-math.expm1(-2*u/T)) / (1+math.exp(-y))
        wa, wb = expit(x)*a, expit(y)*b

        def entropy(z):
            z = abs(z)
            return np.logaddexp(0.0, -z) + z*expit(-z)

        return p*E*np.array([diff, a, b, E*(a+b),
                            p*p/(3*E)*(a+b) if E else 0.0,
                            entropy(x)+entropy(y), wa+wb,
                            (E-u)*wa-(E+u)*wb,
                            (E-u)**2*wa+(E+u)**2*wb])

    moments = _MEASURE*_integrate(integrand, breaks, rtol, 1e-18)
    n, npart, nanti, energy, pressure, entropy, a0, a1, a2 = moments
    if a0 <= 0:
        raise ArithmeticError("Susceptibility underflow; state outside reference domain")
    cv = (a2-a1*a1/a0)/(T*T)
    if cv < -1e-12*max(a2/(T*T), 1e-300):
        raise ArithmeticError("Unresolved negative fixed-net-number heat capacity")
    result = LeptonState(T, mu, m, sign*n,
                         npart if sign > 0 else nanti,
                         nanti if sign > 0 else npart,
                         energy, pressure, entropy, energy-T*entropy,
                         max(0.0, cv), a0/T, sign*a1/(T*T),
                         -sign*a1/(T*a0))
    if not all(math.isfinite(float(v)) for v in result.asdict().values()):
        raise ArithmeticError("Nonfinite finite-temperature lepton response")
    return result


def state_density(temperature_mev, net_density_fm3, mass_mev=M_E, *,
                  rtol=2e-12, tail=60.0):
    """Invert signed net lepton NUMBER density (fm^-3), not electric charge.

    Use mass_mev=M_MU for the independent muon flavor. The scale-aware root
    enforces 1e-9 relative net-number reconstruction; exact zero has its own
    pair-plasma branch. T=0 is not supported by this reference.
    """
    T, _, m, rtol, tail = _inputs(temperature_mev, 0.0, mass_mev, rtol, tail)
    n = _real("net_density_fm3", net_density_fm3)
    if n == 0:
        return state_mu(T, 0.0, m, rtol=rtol, tail=tail)
    target = abs(n)
    pf = HBARC*(3*math.pi**2*target)**(1.0/3.0)
    upper = max(math.hypot(pf, m), T, m)
    if not math.isfinite(upper):
        raise ArithmeticError("Unrepresented net-number root bracket")
    try:
        pair = state_mu(T, 0.0, m, rtol=rtol, tail=tail)
    except ArithmeticError as exc:
        if "Susceptibility underflow" not in str(exc):
            raise
    else:
        estimate = target/pair.dn_dmu_fixed_temperature
        if 0 < estimate < min(T, upper/2):
            upper = 2*estimate
    for _ in range(64):
        if state_mu(T, upper, m, rtol=rtol, tail=tail).net_density >= target:
            break
        upper *= 2
        if not math.isfinite(upper):
            raise ArithmeticError("Unrepresented net-number root bracket")
    else:
        raise ArithmeticError("Net-number inversion failed to bracket")
    fraction = brentq(lambda x: -1.0 if x == 0 else
                      state_mu(T, upper*x, m, rtol=rtol, tail=tail).net_density/target-1,
                      0.0, 1.0, xtol=1e-13, rtol=2e-13)
    result = state_mu(T, math.copysign(upper*fraction, n), m, rtol=rtol, tail=tail)
    if abs(result.net_density/n-1) > 1e-9:
        raise ArithmeticError("Net-number inversion failed relative residual gate")
    return result


def thermal_energy_density(state, *, rtol=2e-12, tail=60.0):
    """Stable e(T,n_net)-e(0,n_net), in MeV fm^-3, for a returned state.

    This is a one-flavor fixed-net-number thermal increment, not matter heat.
    The cold reference retains rest energy; no additional mass/Q term is added.
    """
    if not isinstance(state, LeptonState):
        raise ValueError("A LeptonState returned by this module is required")
    T, _, m, rtol, tail = _inputs(state.temperature, state.chemical_potential,
                                 state.mass, rtol, tail)
    for name, value in state.asdict().items():
        _real(name, value)
    fresh = state_mu(T, state.chemical_potential, m, rtol=rtol, tail=tail)
    if ((state.net_density == 0 and fresh.net_density != 0)
            or (state.net_density != 0 and
                abs(fresh.net_density/state.net_density-1) > 1e-9)):
        raise ValueError("State chemical potential and net number are inconsistent")
    u = abs(state.chemical_potential)
    pf = HBARC*(3*math.pi**2*abs(state.net_density))**(1.0/3.0)
    mu0 = math.hypot(pf, m)
    end = max(0.0, u-m, mu0-m)+tail*T
    if not math.isfinite(end):
        raise ArithmeticError("Unrepresented thermal energy interval")
    breaks = sorted(set([0.0, end] + [k for k in
                        (u-m-tail*T, u-m, u-m+tail*T, mu0-m) if 0 < k < end]))

    def integrand(k):
        E = m+k
        p = math.sqrt(k*(k+2*m))
        x = (E-u)/T
        diff = -expit(x) if E < mu0 else expit(-x)
        return p*E*((E-mu0)*diff+(E+mu0)*expit(-(E+u)/T))

    return float(_MEASURE*_integrate(integrand, breaks, rtol, 1e-20))
