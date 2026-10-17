"""CPU particle-only spin-two relativistic ideal Fermi gas, with fixed mass.

Temperature, total mu and mass are MeV; particle number, entropy and Cv are
fm^-3 (kB=1); energy, kinetic energy, pressure and free energy are MeV fm^-3.
No antiparticles, interactions, neutrality, mass response or JAX AD are supplied.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import numbers

import numpy as np
from scipy.integrate import quad_vec
from scipy.optimize import brentq
from scipy.special import expit

from ..constants import HBARC, M_N, M_P

__all__ = ["ParticleState", "state_mu", "state_density"]
_MEASURE = 1.0/(math.pi**2*HBARC**3)


@dataclass(frozen=True)
class ParticleState:
    """One particle species with mass fixed in every thermodynamic response.

    particle_density is positive particle NUMBER, not particle-minus-anti
    number. kinetic_energy_density directly integrates E-m; total energy and
    mu retain rest mass. No pair-equilibrium completion is implied at high T.
    """
    temperature: float
    mass: float
    chemical_potential: float
    kinetic_chemical_potential: float
    particle_density: float
    kinetic_energy_density: float
    energy_density: float
    pressure: float
    entropy_density: float
    free_energy_density: float
    cv_fixed_particle_density: float
    dn_dmu_fixed_temperature: float
    dn_dtemperature_fixed_mu: float
    dmu_dtemperature_fixed_particle_density: float

    def asdict(self):
        """Return named scalar fields in this module's explicit units."""
        return asdict(self)


def _real(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(name+" must be a finite non-Boolean real scalar")
    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(name+" is not represented as finite float64") from exc
    if not math.isfinite(value):
        raise ValueError(name+" must be finite")
    return value


def _inputs(T, mu, m, rtol, tail):
    T, mu, m, rtol, tail = [_real(name, value) for name, value in
        [("temperature_mev", T), ("chemical_potential_mev", mu),
         ("mass_mev", m), ("rtol", rtol), ("tail", tail)]]
    if T <= 0 or m <= 0 or not 0 < rtol < 1e-4 or not 30 <= tail <= 100:
        raise ValueError("Require T,m>0, 0<rtol<1e-4 and 30<=tail<=100")
    return T, mu, m, rtol, tail


def _integrate(fun, breaks, rtol):
    value = None
    for lo, hi in zip(breaks[:-1], breaks[1:]):
        part, error, info = quad_vec(fun, lo, hi, epsabs=0., epsrel=rtol,
                                     norm="max", limit=500, full_output=True)
        magnitude = float(np.max(abs(part)))
        if (not np.all(np.isfinite(part)) or not math.isfinite(float(error))
                or (not info.success and not
                    (info.status == 2 and error <= rtol*magnitude))):
            raise ArithmeticError("Unresolved particle quadrature")
        value = part if value is None else value+part
    if value is None or not np.all(np.isfinite(value)):
        raise ArithmeticError("Unrepresented particle integral")
    return value


def _restore(value, log_scale):
    if value == 0:
        return 0.0
    try:
        return math.copysign(math.exp(math.log(abs(value))+log_scale), value)
    except OverflowError as exc:
        raise ArithmeticError("Unrepresented particle moment") from exc


def _setup(T, edge, m, tail):
    v = m/T
    end = max(0.0, edge)+tail
    if not all(math.isfinite(x) for x in [v, edge, end]):
        raise ArithmeticError("Unrepresented particle energy interval")
    scale = max(1.0, v, edge)
    log_f0 = float(-np.logaddexp(0.0, -edge))
    log_pref = 3*math.log(T)+2*math.log(scale)+math.log(_MEASURE)+log_f0
    breaks = sorted(set([0.0, end]+[edge+x for x in
                         (-tail, -12., -4., 0., 4., 12., tail)
                         if 0 < edge+x < end]))
    def geometry(x):
        a, b = x/scale, v/scale
        return math.sqrt(a*(a+2*b))*(a+b)
    def occupation(x):
        if edge <= 0:
            return math.exp(-x)*(1+math.exp(edge))/(1+math.exp(edge-x))
        return expit(edge-x)/expit(edge)
    return geometry, occupation, log_f0, log_pref, breaks


def _log_density(T, edge, m, rtol, tail):
    geometry, occupation, _, pref, breaks = _setup(T, edge, m, tail)
    value = float(_integrate(lambda x: geometry(x)*occupation(x), breaks, rtol))
    if value <= 0:
        raise ArithmeticError("Unresolved scaled particle density")
    return math.log(value)+pref


def state_mu(temperature_mev, chemical_potential_mev, mass_mev=M_N, *,
             rtol=2e-12, tail=60.0):
    """Evaluate positive particle populations at T>0 and total mu (MeV).

    The ideal particle-only model is mathematically defined independently of
    a possible equilibrium antiparticle gas, which is omitted, not vetoed.
    No effective-mass rearrangement or weak-current mass choice is supplied.
    """
    T, mu, m, rtol, tail = _inputs(temperature_mev, chemical_potential_mev,
                                  mass_mev, rtol, tail)
    edge = (mu-m)/T
    geometry, occupation, log_f0, pref, breaks = _setup(T, edge, m, tail)
    v = m/T
    kinetic_scale = max(1.0, edge)
    response_scale = max(1.0, -edge)
    def bulk(x):
        measure = geometry(x)*occupation(x)
        pressure = x*(x+2*v)/(3*(v+x))
        return measure*np.array([1.0, x/kinetic_scale, pressure/kinetic_scale])
    number, kinetic, pressure = _integrate(bulk, breaks, rtol)
    def response(x):
        d = x-edge
        w = occupation(x)*expit(d)
        q = abs(d)
        e = math.exp(-q)
        entropy = math.exp(-q-log_f0)*((math.log1p(e)/e if e else 1.)+q/(1+e))
        return geometry(x)*np.array([w, x/kinetic_scale*w,
                                     d/response_scale*w, entropy/response_scale])
    a0, xw, a1, entropy = _integrate(response, breaks, rtol)
    if a0 <= 0:
        raise ArithmeticError("Unresolved particle susceptibility")
    mean_x = kinetic_scale*xw/a0
    # A second centered integral is nonnegative; no subtraction of two large
    # rest/dilution-dominated second moments enters the heat capacity.
    variance = float(_integrate(lambda x: geometry(x)*occupation(x)
                     *expit(x-edge)*(x-mean_x)**2, breaks, rtol))
    density = _restore(number, pref)
    chi = _restore(a0, pref-math.log(T))
    cv = _restore(variance, pref)
    if density <= 0 or chi <= 0 or cv <= 0:
        raise ArithmeticError("Particle density/susceptibility/Cv not represented positive")
    kinetic = _restore(kinetic, pref+math.log(T)+math.log(kinetic_scale))
    pressure = _restore(pressure, pref+math.log(T)+math.log(kinetic_scale))
    entropy = _restore(entropy, pref+math.log(response_scale))
    nT = _restore(a1, pref+math.log(response_scale)-math.log(T))
    result = ParticleState(T, m, mu, mu-m, density, kinetic, m*density+kinetic,
                           pressure, entropy, mu*density-pressure, cv, chi, nT,
                           -response_scale*a1/a0)
    if not all(math.isfinite(float(x)) for x in result.asdict().values()):
        raise ArithmeticError("Nonfinite particle thermodynamics")
    return result


def state_density(temperature_mev, particle_density_fm3, mass_mev=M_N, *,
                  rtol=2e-12, tail=60.0):
    """Invert strictly positive particle NUMBER density in fm^-3.

    Zero/negative density is rejected. A logarithmic residual root retains
    tiny positive populations; final ordinary density must reconstruct to
    1e-9 relative using the returned represented total chemical potential.
    """
    T, _, m, rtol, tail = _inputs(temperature_mev, 0., mass_mev, rtol, tail)
    n = _real("particle_density_fm3", particle_density_fm3)
    if n <= 0:
        raise ValueError("Strictly positive particle NUMBER density required")
    log_n = math.log(n)
    pf = math.exp(math.log(HBARC)+(math.log(3*math.pi**2)+log_n)/3)
    cold_edge = pf*pf/(math.hypot(pf, m)+m)/T
    classical = log_n+3*math.log(HBARC)-math.log(2.)-1.5*(math.log(m)+math.log(T)-math.log(2*math.pi))
    lo, hi = min(-4., classical-4.), max(4., cold_edge+4.)
    fn = lambda edge: _log_density(T, edge, m, rtol, tail)-log_n
    for _ in range(64):
        left, right = fn(lo), fn(hi)
        if left <= 0 <= right:
            break
        if left > 0:
            lo = 2*lo-4
        if right < 0:
            hi = 2*hi+4
    else:
        raise ArithmeticError("Particle density root not bracketed")
    edge = brentq(fn, lo, hi, xtol=2e-13, rtol=2e-13)
    result = state_mu(T, m+T*edge, m, rtol=rtol, tail=tail)
    if abs(result.particle_density/n-1) > 1e-9:
        raise ArithmeticError("Particle density root or total-mu representation residual")
    return result
