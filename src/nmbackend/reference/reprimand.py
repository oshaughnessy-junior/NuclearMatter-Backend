"""RePrimAnd high-accuracy TOV reference (optional; requires ``pyreprimand``).

RePrimAnd (Kastaun et al.; Zenodo 10617510) is our primary *external* reference for
M, R, Lambda, and baryon mass.  This module builds a RePrimAnd barotropic-spline
EOS from an :class:`~nmbackend.eos.base.EOSTable` and runs its TOV branch solver.

Targets the **modern** pyreprimand API (verified against the 1.7 docs:
https://wokast.github.io/RePrimAnd/), with transparent fallbacks to the older API
(the one RIFT's ``EOSManager`` historically used).  The two things that changed:
``tov_acc_simple`` -> ``star_acc_simple`` (now takes two leading bool flags) and
``make_tov_branch_stable`` (num_samp/mgrav_min -> mg_cut_low_rel/mg_cut_low_abs/
gm1_step).  ``make_eos_barotr_spline`` and the sequence accessors are unchanged.

Import-guarded: ``HAVE_REPRIMAND`` is False if ``pyreprimand`` is absent; nothing
here is imported by the package by default.
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

# RePrimAnd's internal baryon mass (kg); units.FORMAL_BARYON_MASS_SI in the docs.
_M_BARYON_REPR_KG = 1.66e-27


# --- version-compatibility shims (modern pyreprimand, legacy fallback) ---
def _pyr_interval(lo, hi):
    for nm in ("range", "interval", "range_t"):
        f = getattr(pyr, nm, None)
        if f is not None:
            return f(lo, hi)
    raise AttributeError("pyreprimand: no interval/range constructor found")


def _pyr_star_acc(acc_tov, acc_deform, minsteps, need_deform=True, need_bulk=False):
    if hasattr(pyr, "star_acc_simple"):
        return pyr.star_acc_simple(need_deform, need_bulk, acc_tov, acc_deform, minsteps)
    return pyr.tov_acc_simple(acc_tov, acc_deform, minsteps)        # legacy


def _pyr_tov_branch(eos, acc, mgrav_min=0.0):
    try:
        return pyr.make_tov_branch_stable(eos, acc, mg_cut_low_rel=0.0,
                                          mg_cut_low_abs=mgrav_min)
    except TypeError:                                              # legacy
        return pyr.make_tov_branch_stable(eos, acc, num_samp=2000,
                                          mgrav_min=mgrav_min)


def build_reprimand_eos(table: EOSTable, n_poly: float = 1.0,
                        pts_per_mag: int = 200):
    """Build a RePrimAnd barotropic-spline EOS object from an EOSTable."""
    if not HAVE_REPRIMAND:
        raise ImportError("pyreprimand not installed; see docs/environment.md")

    n = np.asarray(table.n)                        # fm^-3
    p_mev = np.asarray(table.p_mevfm3)
    eps_mev = np.asarray(table.eps_mevfm3)
    cs2 = np.clip(np.asarray(table.cs2), 1e-8, 1.0)

    rho_si = n * 1e45 * _M_BARYON_REPR_KG          # kg/m^3 (rest-mass density)
    eps_si_J = eps_mev * MEVFM3_TO_PA              # J/m^3
    p_si = p_mev * MEVFM3_TO_PA                    # Pa
    spec_int = np.clip(eps_si_J / (rho_si * C_SI**2) - 1.0, 0.0, None)
    cs = np.sqrt(cs2)

    unew = pyr.units.geom_solar(MSUN_SI)           # geometric G=c=Msun=1 (positional)
    rho_u = rho_si / unew.density
    p_u = p_si / unew.pressure
    rgrho = _pyr_interval(rho_u.min() * 1.0000001, rho_u.max() / 1.0000001)

    # temp / efrac unavailable -> empty lists (per RePrimAnd docs); isentropic EOS
    return pyr.make_eos_barotr_spline(
        rho_u, spec_int, p_u, cs, [], [], True, rgrho, n_poly, unew, pts_per_mag)


def reprimand_sequence(table: EOSTable, n_bins: int = 400, acc: float = 1e-6,
                       mgrav_min: float = 0.3):
    """Return (M_msun, R_km, Lambda, gm1_center) from RePrimAnd's TOV branch."""
    if not HAVE_REPRIMAND:
        raise ImportError("pyreprimand not installed; see docs/environment.md")
    eos = build_reprimand_eos(table)

    star_acc = _pyr_star_acc(acc * 1e-2, acc, 500)
    seq = _pyr_tov_branch(eos, star_acc, mgrav_min=mgrav_min)

    u = seq.units_to_SI
    rg = seq.range_center_gm1
    gm1 = np.linspace(rg.min, rg.max, n_bins)
    M = seq.grav_mass_from_center_gm1(gm1)                     # Msun
    R = seq.circ_radius_from_center_gm1(gm1) * u.length / 1e3  # km
    Lam = seq.lambda_tidal_from_center_gm1(gm1)
    return np.asarray(M), np.asarray(R), np.asarray(Lam), gm1
