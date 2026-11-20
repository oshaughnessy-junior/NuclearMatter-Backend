"""Finite-T nonrelativistic spin-summed density-transition response.

No weak matrix element, opacity normalization or hydro closure is supplied.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from scipy.special import expit, log_expit

HBARC = 197.3269804  # MeV fm


@dataclass(frozen=True)
class PairState:
    T: float  # k_B T, MeV
    mass: float  # common kinetic mass, MeV
    offset_a: float  # rest plus prescribed vector shift, MeV
    offset_b: float
    mu_a: float  # total chemical potential, MeV
    mu_b: float

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.T, self.mass, self.offset_a,
                                              self.offset_b, self.mu_a, self.mu_b)):
            raise ValueError("All state fields must be finite")
        if self.T <= 0 or self.mass <= 0:
            raise ValueError("Require T>0 and common kinetic mass>0")

    @property
    def kinetic_mu_a(self):
        return self.mu_a-self.offset_a

    @property
    def kinetic_mu_b(self):
        return self.mu_b-self.offset_b

    def reverse(self):
        return PairState(self.T, self.mass, self.offset_b, self.offset_a, self.mu_b, self.mu_a)


def _log_integral(a: float, z: float):
    """log of integral_0^infty dx f(x-a)[1-f(x-a+z)].

    Exact elementary expression except a controlled small-z series and
    exponentially accurate log-tail branch. No physical threshold smoothing.
    """
    if z < 0:
        return z + _log_integral(a-z, -z)
    f = expit(a)
    if z < 1e-5:
        correction = .5*z*f + z*z*f*(4*f-3)/12
        return float(log_expit(a) + math.log1p(correction))
    denominator = -math.expm1(-z)
    if a < -40:
        # Neglected relative terms are < exp(-40), below float64 target.
        return a-math.log1p(math.exp(a-z))
    if a <= 0:
        difference = math.log1p(math.exp(a)*denominator/(1+math.exp(a-z)))
    elif a-z >= 0:
        difference = z + math.log1p(math.exp(-a))-math.log1p(math.exp(-(a-z)))
    else:
        difference = float(math.log1p(math.exp(-a))+a-math.log1p(math.exp(a-z)))
    return math.log(difference)-math.log(denominator)


def kinematic_lower_energy(state: PairState, q: float, omega: float):
    """Minimum initial kinetic energy, MeV; q=|momentum transfer| in MeV."""
    if not math.isfinite(q) or not math.isfinite(omega) or q <= 0:
        raise ValueError("Require finite q>0 and finite omega; q=0 is distributional")
    w = omega-(state.offset_b-state.offset_a)
    pz = state.mass*w/q-q/2
    return pz*pz/(2*state.mass)


def log_response(state: PairState, q: float, omega: float):
    """log S_ab(q,omega); S in fm^-3 MeV^-1, including spin=2 and 2pi delta.

    omega is energy transferred INTO nucleons, with p_b=p_a+q.
    Finite-T support is all real omega in this nonrelativistic ansatz.
    """
    emin = kinematic_lower_energy(state, q, omega)
    a = (state.kinetic_mu_a-emin)/state.T
    z = (omega+state.mu_a-state.mu_b)/state.T
    return (2*math.log(state.mass)-math.log(math.pi*q)-3*math.log(HBARC)
            +math.log(state.T)+_log_integral(a, z))


def response(state: PairState, q: float, omega: float):
    """Positive response; use log_response for float64-underflow tails."""
    return math.exp(log_response(state, q, omega))


def cold_response(mass, offset_a, offset_b, mu_a, mu_b, q, omega):
    """Exact T=0 interval length, with Pauli blocking and recoil."""
    if not all(math.isfinite(v) for v in (mass, offset_a, offset_b, mu_a, mu_b, q, omega)):
        raise ValueError("All arguments must be finite")
    if mass <= 0 or q <= 0:
        raise ValueError("Require positive mass and q")
    w = omega-(offset_b-offset_a)
    emin = (mass*w/q-q/2)**2/(2*mass)
    interval = max(0., mu_a-offset_a-max(emin, mu_b-offset_b-w))
    return mass*mass/(math.pi*q*HBARC**3)*interval
