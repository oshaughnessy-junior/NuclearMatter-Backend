"""Independent radius-domain TOV + tidal reference solver (NumPy/SciPy).

This is deliberately a *different* method from ``nmbackend.tov.solver`` so it can
serve as an in-process cross-check:

    * integration variable: **radius r** (not pseudo-enthalpy h),
    * stepper: **SciPy adaptive RK45** (not fixed-step RK4),
    * tidal sector: the second-order **(H, beta)** system (Hinderer 2008), not the
      first-order y-Riccati used by the production solver.

If the two agree to high precision on a smooth EOS, both formulations are
corroborated.  Where they *disagree* (e.g. across a strong phase transition), the
adaptive RK45 here is the more trustworthy of the two and flags work for the
production stepper.  This is **not** differentiable and is not in the hot path.

Geometric units: r, m in km; eps, p in km^-2 (consuming an ``EOSTable``).
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from scipy.integrate import solve_ivp

from ..eos.base import EOSTable
from ..constants import GMSUN_KM

_FOURPI = 4.0 * math.pi


class RefPoint(NamedTuple):
    h_c: float
    R_km: float
    M_km: float
    M_msun: float
    C: float
    y: float
    k2: float
    Lambda: float


class _EOSInterp:
    """log-space NumPy interpolants eps(p), cs2(p) from an EOSTable (geometric)."""

    def __init__(self, table: EOSTable):
        self.h = np.asarray(table.h, dtype=float)
        self.eps = np.asarray(table.eps, dtype=float)
        self.p = np.asarray(table.p, dtype=float)
        self.cs2 = np.clip(np.asarray(table.cs2, dtype=float), 1e-8, 1.0)
        self._logp = np.log(self.p)
        self._logeps = np.log(self.eps)
        self.p_min = float(self.p[0])
        self.p_max = float(self.p[-1])

    def eps_of_p(self, p):
        lp = np.log(np.clip(p, self.p_min, self.p_max))
        return np.exp(np.interp(lp, self._logp, self._logeps))

    def cs2_of_p(self, p):
        lp = np.log(np.clip(p, self.p_min, self.p_max))
        return np.interp(lp, self._logp, self.cs2)

    def central_from_hc(self, h_c):
        eps_c = math.exp(np.interp(h_c, self.h, self._logeps))
        p_c = math.exp(np.interp(h_c, self.h, self._logp))
        return eps_c, p_c


def _k2_from_y_C(y, C):
    onemtwoC = 1.0 - 2.0 * C
    num = (8.0 / 5.0) * C**5 * onemtwoC**2 * (2.0 + 2.0 * C * (y - 1.0) - y)
    den = (
        2.0 * C * (6.0 - 3.0 * y + 3.0 * C * (5.0 * y - 8.0))
        + 4.0 * C**3 * (13.0 - 11.0 * y + C * (3.0 * y - 2.0)
                        + 2.0 * C**2 * (1.0 + y))
        + 3.0 * onemtwoC**2 * (2.0 - y + 2.0 * C * (y - 1.0)) * math.log(onemtwoC)
    )
    return num / den


def solve_reference(table: EOSTable, h_c: float, rtol=1e-8, atol=1e-10,
                    r0=1e-4, r_max=60.0):
    """Reference (R, M, k2, Lambda) at central pseudo-enthalpy h_c."""
    eos = _EOSInterp(table)
    eps_c, p_c = eos.central_from_hc(h_c)

    # central series: m ~ 4/3 pi eps r^3 ; H ~ a0 r^2 ; beta = dH/dr ~ 2 a0 r.
    m0 = (4.0 / 3.0) * math.pi * eps_c * r0**3
    H0 = r0**2
    beta0 = 2.0 * r0
    p_surface = eos.p_min

    def rhs(r, s):
        m, p, H, beta = s
        p = max(p, p_surface)
        eps = float(eos.eps_of_p(p))
        cs2 = float(eos.cs2_of_p(p))
        one_minus = 1.0 - 2.0 * m / r
        dm = _FOURPI * r**2 * eps
        dp = -(eps + p) * (m + _FOURPI * r**3 * p) / (r**2 * one_minus)
        dH = beta
        # Hinderer 2008 (H, beta) second-order tidal system
        term_a = (2.0 / one_minus) * (
            -2.0 * math.pi * (5.0 * eps + 9.0 * p + (eps + p) / cs2)
            + 3.0 / r**2
            + 2.0 / one_minus * ((m + _FOURPI * r**3 * p) / r**2) ** 2
        )
        term_b = (2.0 * beta / r) / one_minus * (
            -1.0 + m / r + 2.0 * math.pi * r**2 * (eps - p)
        )
        dbeta = term_a * H + term_b
        return [dm, dp, dH, dbeta]

    def hit_surface(r, s):
        return s[1] - p_surface
    hit_surface.terminal = True
    hit_surface.direction = -1

    sol = solve_ivp(rhs, (r0, r_max), [m0, p_c, H0, beta0],
                    method="RK45", rtol=rtol, atol=atol, events=hit_surface,
                    dense_output=False, max_step=r_max)
    R = float(sol.t[-1])
    m, _, H, beta = sol.y[:, -1]
    M = float(m)
    C = M / R
    y = R * beta / H
    k2 = _k2_from_y_C(y, C)
    Lam = (2.0 / 3.0) * k2 / C**5
    return RefPoint(h_c=float(h_c), R_km=R, M_km=M, M_msun=M / GMSUN_KM,
                    C=C, y=float(y), k2=float(k2), Lambda=float(Lam))


def reference_sequence(table: EOSTable, h_c_array):
    return [solve_reference(table, float(hc)) for hc in np.asarray(h_c_array)]
