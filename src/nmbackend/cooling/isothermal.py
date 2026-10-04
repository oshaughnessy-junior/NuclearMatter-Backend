"""Isothermal-interior cooling curve from an ``IsothermalStar`` table.

With a uniform redshifted interior temperature T, NSCool's energy
equation summed over zones is C(T) dT/dt = -[L_nu(T) - L_heat(T) + L_gamma(T)],
so the age at temperature T is

    t(T) = integral_T^T0 C / (L_nu - L_heat + L_gamma) dT'.

With ``blanket=True`` the star is isothermal only below density rho_b:
the layer from NSCool's outer boundary to rho_b carries L_gamma by
conduction (``IsothermalStar.T_in``), so the interior at T_in(T_b) has
C(T_in) and L_nu(T_in) while the surface radiates L_gamma(T_b).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .runner import IsothermalStar

__all__ = ["IsothermalCurve", "isothermal_curve"]

YEAR_S = 3.15576e7


class IsothermalCurve(NamedTuple):
    t_yr: np.ndarray       # age, increasing
    T: np.ndarray          # redshifted interior temperature
    Teff_inf: np.ndarray
    L_gamma_inf: np.ndarray
    L_nu_inf: np.ndarray

    def at(self, t_yr, quantity: str = "Teff_inf") -> np.ndarray:
        """Log-log interpolation at finite positive ages within curve support.

        Raises ValueError rather than returning an unsupported endpoint value.
        """
        ages = np.asarray(self.t_yr, float)
        query = np.asarray(t_yr, float)
        if (ages.ndim != 1 or ages.size == 0 or not np.all(np.isfinite(ages))
                or np.any(ages <= 0) or np.any(np.diff(ages) <= 0)):
            raise ValueError("curve ages must be finite, positive and increasing")
        if not np.all(np.isfinite(query)) or np.any(query <= 0):
            raise ValueError("queried ages must be finite and positive")
        if np.any(query < ages[0]) or np.any(query > ages[-1]):
            raise ValueError(f"queried age outside curve support [{ages[0]:g}, {ages[-1]:g}] yr")
        y = getattr(self, quantity)
        return np.exp(np.interp(np.log(query), np.log(ages), np.log(y)))


def isothermal_curve(star: IsothermalStar, blanket: bool = False) -> IsothermalCurve:
    """Integrate t down the grid, with t = 0 at the top of the grid.

    ``T`` of the result is the interior temperature (T_in with
    ``blanket``).  The integrand C T / L is integrated in ln T by the
    trapezoid rule on NSCool's grid; refine the grid
    (``NSCool.isothermal(T_grid=...)``) to check convergence.
    """
    o = np.argsort(star.T)[::-1]                # decreasing T (= T_b)
    lnT = np.log(star.T[o])
    C, L_nu, L_heat = star.C[o], star.L_nu[o], star.L_heat[o]
    if blanket:
        lnTi = np.log(star.T_in[o])
        if np.any(np.diff(lnTi) >= 0):
            raise ValueError("T_in is not monotonic in T_b")
        C, L_nu, L_heat = (np.exp(np.interp(-lnTi, -lnT, np.log(np.maximum(y, 1e-300))))
                           for y in (C, L_nu, L_heat))
        lnT = lnTi
    L = L_nu - L_heat + star.L_gamma[o]
    if np.any(L <= 0):
        raise ValueError("net luminosity is not positive on the grid")
    f = C / L * np.exp(lnT)
    t = np.cumsum(0.5 * (f[1:] + f[:-1]) * -np.diff(lnT))
    return IsothermalCurve(t / YEAR_S, np.exp(lnT[1:]), star.Teff_inf[o][1:],
                           star.L_gamma[o][1:], L_nu[1:])
