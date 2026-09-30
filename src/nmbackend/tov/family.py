"""M-R-Lambda sequences, stable-branch detection, and dM/dh_c, dh_c/dM.

The planning notes ask us to parameterize stellar sequences by *central* variable
(pseudo-enthalpy / entropy) rather than mass, and to expose dm/dh and dh/dm so a
user can do inference "as if parameterized by mass" while we integrate in h.  This
module builds the sequence by ``vmap``-ing the TOV solver over central enthalpy,
computes dM/dh_c by AD, splits stable (dM/dh_c > 0) from unstable branches, and
provides interpolators on the stable branch.
"""
from __future__ import annotations

from typing import NamedTuple
import jax
import jax.numpy as jnp

from .solver import solve_structure, TOVPoint
from ..eos.base import EOSTable


class Family(NamedTuple):
    h_c: jnp.ndarray
    M_msun: jnp.ndarray
    R_km: jnp.ndarray
    Lambda: jnp.ndarray
    C: jnp.ndarray
    dM_dhc: jnp.ndarray      # dM[Msun]/dh_c along the sequence
    stable: jnp.ndarray      # bool mask: dM/dh_c > 0


def _mass_and_slope(table, hc):
    return jax.jvp(lambda h: solve_structure(table, h, tidal=False, rtol=1e-6,
                                             atol=1e-8).M_msun,
                   (hc,), (jnp.ones_like(hc),))


# jitted at module level with the table as an argument: one compile per
# table length, not one per table
_probe_masses = jax.jit(jax.vmap(_mass_and_slope, in_axes=(None, 0)))


def default_hc_grid(table: EOSTable, n_models: int = 120, frac_lo: float = 0.02,
                    frac_hi: float = 0.92, n_c_min: float = 0.07,
                    n_c_probe: float = 0.25, n_probe: int = 16):
    """A central-enthalpy grid from the minimum-mass star to near-maximum
    central density.

    The search starts at the larger of ``frac_lo`` h_max and h(n = ``n_c_min``
    fm^-3); with a crust table, frac_lo h_max alone falls deep in the crust.
    The minimum-mass star can still lie above n_c_min (n_c = 0.093 fm^-3 for
    the MetaModel crust), so ``n_probe`` structure-only solves up to
    h(n = ``n_c_probe``) locate it and the grid starts at the first probe on
    the rising branch.  ``n_probe=0`` skips the probe."""
    hmax = table.h[-1]
    h_lo = jnp.maximum(frac_lo * hmax,
                       jnp.interp(jnp.log(n_c_min), jnp.log(table.n), table.h))
    if n_probe:
        h_top = jnp.minimum(
            frac_hi * hmax,
            jnp.interp(jnp.log(n_c_probe), jnp.log(table.n), table.h))
        probe = jnp.linspace(h_lo, jnp.maximum(h_top, h_lo), n_probe)
        M, dM = _probe_masses(table, probe)
        i = jnp.argmin(M)
        h_lo = jnp.where(dM[i] > 0.0, probe[i],
                         probe[jnp.minimum(i + 1, n_probe - 1)])
    return jnp.linspace(h_lo, frac_hi * hmax, n_models)


def build_family(table: EOSTable, h_c_grid=None, *, backend: str = "diffrax",
                 rtol: float = 1e-7, atol: float = 1e-9,
                 n_steps: int = 2000) -> Family:
    if h_c_grid is None:
        h_c_grid = default_hc_grid(table)

    kw = dict(backend=backend, rtol=rtol, atol=atol, n_steps=n_steps)

    def M_of_hc(hc):
        return solve_structure(table, hc, **kw).M_msun

    solve_all = jax.vmap(lambda hc: solve_structure(table, hc, **kw))
    pts: TOVPoint = solve_all(h_c_grid)

    # forward-mode along the sequence (clean through the tidal variable)
    dM_dhc = jax.vmap(jax.jacfwd(M_of_hc))(h_c_grid)
    # the neutron-star branch runs from the minimum-mass star to M_max; with a
    # crust table the lowest h_c can lie below the minimum-mass star, on the
    # extended (R ~ 10^3 km) configurations that also have dM/dh_c > 0
    rising = dM_dhc > 0.0
    order = jnp.arange(h_c_grid.shape[0])
    i_max = jnp.argmax(jnp.where(rising, pts.M_msun, -jnp.inf))
    i_min = jnp.argmin(jnp.where(order <= i_max, pts.M_msun, jnp.inf))
    stable = rising & (order >= i_min)

    return Family(
        h_c=h_c_grid, M_msun=pts.M_msun, R_km=pts.R_km, Lambda=pts.Lambda,
        C=pts.C, dM_dhc=dM_dhc, stable=stable,
    )


def m_max(family: Family):
    """Maximum gravitational mass (Msun) over the (stable part of the) sequence."""
    M = jnp.where(family.stable, family.M_msun, -jnp.inf)
    return jnp.max(M)


def dhc_dM(family: Family):
    """Inverse Jacobian dh_c/dM on the sequence (NaN where dM/dh_c ~ 0)."""
    # convert dM[Msun]/dh_c into dh_c/dM
    return 1.0 / family.dM_dhc


_BIG_M = 1.0e3  # finite "above the branch" mass sentinel (avoids 0*inf -> nan in AD)


def _stable_branch_sort(family: Family):
    """Indices sorting the masses on the stable rising branch (minimum-mass
    star to M_max) to the front.

    Off-branch masses are pushed to a large *finite* value so reverse-mode AD does
    not hit 0*inf -> nan.  Returns (M_sorted, sort_idx, idx_max).
    """
    M = family.M_msun
    idx_max = jnp.argmax(jnp.where(family.stable, M, -jnp.inf))
    idx_min = jnp.argmax(family.stable)          # first stable star (minimum mass)
    order = jnp.arange(M.shape[0])
    on_branch = (order >= idx_min) & (order <= idx_max)
    Mb = jnp.where(on_branch, M, _BIG_M)
    srt = jnp.argsort(Mb)
    return Mb[srt], srt, idx_max


def lambda_of_mass(family: Family, m_msun):
    """Interpolate Lambda(M) on the stable rising branch.

    Above M_max the off-branch sentinel makes the interpolant HOLD near
    Lambda(M_max) (reaching 0 only beyond the sentinel mass) -- callers must
    range-guard m <= m_max(family) themselves.
    """
    Ms, srt, _ = _stable_branch_sort(family)
    Ls = family.Lambda[srt]
    return jnp.interp(m_msun, Ms, Ls, left=Ls[0], right=0.0)


def radius_of_mass(family: Family, m_msun):
    """Interpolate R(M) on the stable rising branch."""
    Ms, srt, _ = _stable_branch_sort(family)
    Rs = family.R_km[srt]
    return jnp.interp(m_msun, Ms, Rs, left=Rs[0], right=Rs[-1])
