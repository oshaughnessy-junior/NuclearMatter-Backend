"""RePrimAnd high-accuracy TOV reference (optional; requires ``pyreprimand``).

RePrimAnd (Kastaun et al.; Zenodo 10617510) is our primary *external* reference for
M, R, Lambda, and baryon mass.  This module builds a RePrimAnd barotropic-spline
EOS from an :class:`~nmbackend.eos.base.EOSTable` and runs its TOV branch solver.

API CAVEAT (read this):
    RePrimAnd's Python API has changed across releases, and RIFT's
    ``EOSManager.EOSReprimand`` targets an **older** interface.  The calls here
    mirror that known-old interface (``pyr.units.geom_solar``, ``pyr.range``,
    ``pyr.make_eos_barotr_spline``, ``pyr.tov_acc_simple``,
    ``pyr.make_tov_branch_stable``, ``seq.*_from_center_gm1``).  Lines marked
    ``# VERIFY`` are the most likely to need adjustment for your installed
    ``pyreprimand`` version -- check ``help(pyr.make_eos_barotr_spline)`` and the
    RePrimAnd docs (https://wokast.github.io/RePrimAnd/).  This file is a scaffold
    you run in the validation environment (see docs/environment.md), not something
    exercised in the core test suite.

It is import-guarded: ``HAVE_REPRIMAND`` is False if ``pyreprimand`` is absent, and
nothing here is imported by the package by default.
"""
from __future__ import annotations

import numpy as np

from ..constants import MEVFM3_TO_PA, MSUN_SI, C_SI
from ..eos.base import EOSTable

try:
    import pyreprimand as pyr
    HAVE_REPRIMAND = True
except Exception:  # pragma: no cover
    pyr = None
    HAVE_REPRIMAND = False

# RePrimAnd's internal baryon mass (kg); see RIFT EOSManager notes.
_M_BARYON_REPR_KG = 1.66e-27


def build_reprimand_eos(table: EOSTable, n_poly: float = 1.0,
                        pts_per_mag: int = 200):
    """Build a RePrimAnd barotropic-spline EOS object from an EOSTable."""
    if not HAVE_REPRIMAND:
        raise ImportError("pyreprimand not installed; see docs/environment.md")

    n = np.asarray(table.n)                       # fm^-3
    p_mev = np.asarray(table.p_mevfm3)
    eps_mev = np.asarray(table.eps_mevfm3)
    cs2 = np.clip(np.asarray(table.cs2), 1e-8, 1.0)

    # rest-mass density (SI) and total energy density (SI, J/m^3)
    rho_si = n * 1e45 * _M_BARYON_REPR_KG          # kg/m^3
    eps_si_J = eps_mev * MEVFM3_TO_PA              # J/m^3
    p_si = p_mev * MEVFM3_TO_PA                    # Pa
    spec_int = eps_si_J / (rho_si * C_SI**2) - 1.0  # dimensionless
    spec_int = np.clip(spec_int, 0.0, None)
    cs = np.sqrt(cs2)

    unew = pyr.units.geom_solar(msun_si=MSUN_SI)              # VERIFY units ctor
    rho_u = rho_si / unew.density                            # VERIFY .density
    p_u = p_si / unew.pressure                               # VERIFY .pressure
    rgrho = pyr.range(rho_u.min() * 1.0000001,               # VERIFY pyr.range
                      rho_u.max() / 1.0000001)

    temp = np.zeros_like(rho_u)
    efrac = np.full_like(rho_u, 0.1)
    isentropic = True
    eps_0 = 0.0
    # VERIFY: argument order/identity of make_eos_barotr_spline for your version
    return pyr.make_eos_barotr_spline(
        rho_u, spec_int, p_u, cs, temp, efrac, isentropic,
        rgrho, n_poly, unew, pts_per_mag,
    )


def reprimand_sequence(table: EOSTable, n_bins: int = 400, acc: float = 1e-6):
    """Return (M_msun, R_km, Lambda, gm1_center) from RePrimAnd's TOV branch."""
    if not HAVE_REPRIMAND:
        raise ImportError("pyreprimand not installed; see docs/environment.md")
    eos = build_reprimand_eos(table)

    tov_acc = pyr.tov_acc_simple(acc * 1e-2, acc, 500)        # VERIFY signature
    seq = pyr.make_tov_branch_stable(eos, tov_acc, num_samp=2000, mgrav_min=0.3)

    u = seq.units_to_SI
    rg = seq.range_center_gm1
    gm1 = np.linspace(rg.min, rg.max, n_bins)
    M = seq.grav_mass_from_center_gm1(gm1)                    # Msun
    R = seq.circ_radius_from_center_gm1(gm1) * u.length / 1e3  # km
    Lam = seq.lambda_tidal_from_center_gm1(gm1)
    return np.asarray(M), np.asarray(R), np.asarray(Lam), gm1
