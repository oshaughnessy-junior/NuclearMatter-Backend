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


def default_hc_grid(table: EOSTable, n_models: int = 120, frac_lo: float = 0.02,
                    frac_hi: float = 0.92):
    """A central-enthalpy grid spanning low-mass to near-maximum central density."""
    hmax = table.h[-1]
    return jnp.linspace(frac_lo * hmax, frac_hi * hmax, n_models)


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
    stable = dM_dhc > 0.0

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
    """Indices sorting the masses on the stable rising branch to the front.

    Off-branch masses are pushed to a large *finite* value so reverse-mode AD does
    not hit 0*inf -> nan.  Returns (M_sorted, sort_idx, idx_max).
    """
    M = family.M_msun
    idx_max = jnp.argmax(jnp.where(family.stable, M, -jnp.inf))
    order = jnp.arange(M.shape[0])
    on_branch = order <= idx_max
    Mb = jnp.where(on_branch, M, _BIG_M)
    srt = jnp.argsort(Mb)
    return Mb[srt], srt, idx_max


def lambda_of_mass(family: Family, m_msun):
    """Interpolate Lambda(M) on the stable rising branch (0 above M_max)."""
    Ms, srt, _ = _stable_branch_sort(family)
    Ls = family.Lambda[srt]
    return jnp.interp(m_msun, Ms, Ls, left=Ls[0], right=0.0)


def radius_of_mass(family: Family, m_msun):
    """Interpolate R(M) on the stable rising branch."""
    Ms, srt, _ = _stable_branch_sort(family)
    Rs = family.R_km[srt]
    return jnp.interp(m_msun, Ms, Rs, left=Rs[0], right=Rs[-1])
