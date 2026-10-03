"""Heat capacity per volume (erg cm^-3 K^-1), as NSCool's ``spec_heat.f``.

Degenerate fermions: C = (pi^2/3) N(0) k_B^2 T with N(0) = m* p_F / pi^2
(``cvt_deg``; leptons use m* = sqrt(m^2 + p_F^2)).  Superfluid reduction
of Levenfish & Yakovlev 1994 (``r_1s0``, ``r_3p2``).  Ions: NSCool's
``cvion`` (liquid fit, then a Debye-table solid).  Most of spec_heat.f's
literals are single precision; they are kept so.

NSCool departures kept here: the proton C is not the BCS jump at Tc; it
falls linearly from the reduced value just below Tc to the normal value
at 1.1 Tc.  Neutrons have no such ramp.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .superfluid import f32, fexp

__all__ = ["cv_coeff", "cv_neutron", "cv_proton", "cv_ion", "R_1S0", "R_3P2"]

_PI = 3.14159265
_KB, _MEV = 1.38e-16, 1.602e-6
_PI4 = f32(3.14159265)                 # spec_heat.f: parameter (pi = 3.14159265)


def cv_coeff(kf, m):
    """cvt_deg: C/T (erg cm^-3 K^-2) for k_F in fm^-1 and m in MeV."""
    pf = kf * 197.0
    n0 = 2.0 * m * pf / 2.0 / _PI ** 2
    return jnp.where(pf == 0.0, 0.0, _PI ** 2 / 3.0 * n0) * _KB ** 2 / _MEV / 197.0 ** 3 * 1e39


def lepton_cv_coeff(kf, m0):
    """C/T for electrons (m0 = 0.511 MeV) or muons (105 MeV)."""
    pf = kf * 197.0
    return cv_coeff(kf, jnp.sqrt(m0 ** 2 + pf ** 2))


_A = [f32(c) for c in (1.456, 0.157, 1.764, 0.4186, 1.007, 0.5010,
                       5.596, 8.424, 0.6893, 0.790, 0.03983, 1.934)]


def _u1(t):
    return jnp.sqrt(1.0 - t) * (_A[0] - _A[1] / jnp.sqrt(t) + _A[2] / t)


def R_1S0(u):
    return (_A[3] + jnp.sqrt(_A[4] ** 2 + (_A[5] * u) ** 2)) ** 2.5 * fexp(_A[0] - jnp.sqrt(_A[0] ** 2 + u ** 2))


def _u3(t):
    return jnp.sqrt(1.0 - t) * (_A[6] + _A[7] / t)


def R_3P2(u):
    return ((_A[8] + jnp.sqrt(_A[9] ** 2 + (_A[10] * u) ** 2)) ** 2
            * fexp(_A[11] - jnp.sqrt(_A[11] ** 2 + u ** 2 / (16.0 * _PI4))))


_T0MAX = 0.999999999999


def cv_neutron(T, cvn, tcn, i, isf):
    """Neutron C (3P2 reduction for i <= isf, 1S0 above); NSCool applies it
    only at rho >= rhodrip, which the caller masks."""
    m = (T < tcn) & (tcn > 0)
    t0 = jnp.where(m, jnp.minimum(_T0MAX, T / jnp.where(m, tcn, 1.0)), 0.5)
    r = jnp.where(i <= isf, R_3P2(_u3(t0)), R_1S0(_u1(t0)))
    return cvn * T * jnp.where(m, r, 1.0)


def cv_proton(T, cvp, tcp, raise_=1.1):
    """Proton C with NSCool's ramp from Tc to 1.1 Tc."""
    m = (T < raise_ * tcp) & (tcp > 0)
    tcs = jnp.where(m, tcp, 1.0)
    t0 = jnp.where(m, jnp.minimum(_T0MAX, T / tcs), 0.5)
    r = R_1S0(_u1(t0))
    w1 = (raise_ * tcs - T) / ((raise_ - 1.0) * tcs)
    r = jnp.where(T > tcp, w1 * r + (1.0 - w1), r)
    return cvp * T * jnp.where(m, r, 1.0)


_BCV, _CCV, _DCV, _HCV, _CTE = (f32(c) for c in (.95043, .18956, -.81487, 3225.0, 1.417e2))
_CV0 = np.asarray([0.0, 2.956, 2.829, 2.633, 2.389, 2.118, 1.840, 1.572,
                   1.323, 1.102, 0.909, 0.745, 0.609, 0.496, 0.404], np.float32).astype(float)
_G, _NA, _KBS, _THIRD = f32(2.273e5), f32(6.022e23), f32(1.38e-16), f32(1.0 / 3.0)


def cv_ion(T, rho, A, Z):
    """cvion: ions per cm^3 with A = a_cell (nucleons per cell), Z per nucleus.

    Zones without nuclei (A or Z = 0, as in the core) give 0 with finite
    gradients.  In the 178 < Gamma < 210 blend NSCool reads past its Debye
    table when delta > 7e20 (low Z only, not reached by our crusts); here
    the table is clamped at its last entry.
    """
    ok = (A > 0) & (Z > 0)
    A, Z = jnp.where(ok, A, 1.0), jnp.where(ok, Z, 1.0)
    return jnp.where(ok, _cv_ion(T, rho, A, Z), 0.0)


def _cv_ion(T, rho, A, Z):
    gamma = _G * Z ** 2 * (rho / A) ** _THIRD / T
    a1 = jnp.where(rho >= f32(4.3e11), 3.0 * Z, A)       # cvion's own drip density
    nk = _KBS * _NA * rho / A
    delta = 1.0 / T * Z * jnp.sqrt(rho / (a1 * A)) * _NA
    g = jnp.maximum(gamma, 1e-30)
    liquid = nk * (.75 * _BCV * g ** .25 + 1.25 * _CCV / g ** .25 + _DCV + 1.5)
    gas = 1.5 * nk
    solid0 = 1.5 + 3.0 * _HCV / g ** 2 + 1.5             # cv0(0), set at each call
    x = jnp.minimum(delta * f32(2e-20), 14.0)
    i1 = jnp.clip(jnp.floor(x).astype(int), 0, 13)
    cv0 = lambda j: jnp.where(j == 0, solid0, jnp.asarray(_CV0)[j])
    table = nk * (cv0(i1) + (x - i1) * (cv0(i1 + 1) - cv0(i1)))
    harm = nk * solid0
    debye = nk * _CTE / (delta * f32(1e-20)) ** 3
    blend = (gamma - 178.0) / 32.0 * table + (210.0 - gamma) / 32.0 * harm
    p1, p2 = f32(.1), f32(.2)
    return jnp.select(
        [gamma <= p1, gamma <= p2, gamma <= 178.0,
         (gamma <= 210.0) & (delta >= f32(1e19)), delta <= f32(1e19),
         delta < f32(7e20)],
        [gas, (gamma - p1) / p1 * liquid + (p2 - gamma) / p1 * gas, liquid,
         blend, harm, table], debye)
