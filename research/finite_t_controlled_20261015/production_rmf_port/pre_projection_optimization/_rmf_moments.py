"""Private spin-two pair-gas/scalar response moments, all in internal fm units."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from scipy.integrate import quad_vec
from scipy.special import expit

@dataclass(frozen=True)
class Gas:
    density: float
    scalar: float
    energy: float
    pressure: float
    entropy: float
    chi: float
    scalar_nu: float
    scalar_mass: float
    n_temperature: float
    scalar_temperature: float
    s_temperature: float
    antiparticle_density: object
    log_antiparticle_density: float


def moments(t, nu, mass, *, rtol=2e-11, tail=60.):
    """Number is NET; scalar/energy/entropy add thermal antiparticles, no sea.

    t,nu,mass are fm^-1. Partial responses hold the other internal variables fixed.
    """
    if not all(math.isfinite(x) for x in (t, nu, mass, rtol, tail)) or t <= 0 or mass <= 0:
        raise ValueError("Positive finite internal temperature and mass required")
    end = max(0., abs(nu)-mass)+tail*t
    breaks = sorted(set([0., end]+[abs(nu)-mass+a*t for a in
        (-tail, -12., -4., 0., 4., 12., tail) if 0 < abs(nu)-mass+a*t < end]))
    log_b0 = float(-np.logaddexp(0., (mass+nu)/t))
    z0 = -(mass+nu)/t
    def integrand(k):
        e = mass+k
        p = math.sqrt(k*(k+2*mass))
        jac = p*e/math.pi**2
        x, y = (e-nu)/t, (e+nu)/t
        a, b = expit(-x), expit(-y)
        wa, wb = a*expit(x), b*expit(y)
        def entropy(z):
            az = abs(z)
            return np.logaddexp(0., -az)+az*expit(-az)
        # Scaled anti component remains resolved when its ordinary value underflows.
        logratio = (-k/t+math.log1p(math.exp(z0))-math.log1p(math.exp(z0-k/t))) if z0 <= 0 else float(-np.logaddexp(0., y))-log_b0
        scaled_b = math.exp(logratio)
        return jac*np.array([a-b, mass/e*(a+b), e*(a+b), p*p/(3*e)*(a+b),
            entropy(x)+entropy(y), (wa+wb)/t, mass/e*(wa-wb)/t,
            p*p/e**3*(a+b)-(mass/e)**2*(wa+wb)/t,
            (x*wa-y*wb)/t, mass/e*(x*wa+y*wb)/t,
            (x*x*wa+y*y*wb)/t, scaled_b])
    result = np.zeros(12)
    for lo, hi in zip(breaks[:-1], breaks[1:]):
        v, error, info = quad_vec(integrand, lo, hi, epsabs=0., epsrel=rtol,
                                  norm="max", limit=400, full_output=True)
        if not np.all(np.isfinite(v)) or not math.isfinite(float(error)) or (
            not info.success and not (info.status == 2 and error <= rtol*np.max(abs(v)))):
            raise ArithmeticError("Unresolved RMF pair quadrature")
        result += v
    if not np.all(np.isfinite(result)) or min(result[1], result[2], result[4], result[5], result[11]) <= 0:
        raise ArithmeticError("Unrepresented positive RMF gas moments")
    loganti = math.log(result[11])+log_b0
    anti = math.exp(loganti)
    return Gas(*map(float, result[:11]), anti if anti > 0 else None, float(loganti))
