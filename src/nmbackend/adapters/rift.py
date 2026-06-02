"""RIFT ``EOSConcrete``-compatible adapter.

RIFT's ``EOSManager.EOSConcrete`` objects are consumed downstream mainly through

    * ``lambda_from_m(m)``           -> tidal deformability at gravitational mass m
    * ``lambda_from_m_vector(marr)``
    * ``pressure_density_on_grid(logrho_grid, ...)`` -> log10 p(SI) vs log10 rho(SI)
    * ``mMaxMsun``                   -> maximum mass (prior bound)
    * ``name``                       -> identifier

``EOSConcreteNMB`` provides exactly this surface, backed by the differentiable
JAX TOV family, so a NuclearMatter-Backend EOS drops into "normal" RIFT (per-EOS
evidence) and hence the tabular hyperpipeline.  It does **not** import lalsuite;
``self.eos``/``self.eos_fam`` are exposed (table / ``None``) for duck-typing.

Accepts mass either in solar masses (~1.4) or SI kg (>1e10), matching RIFT.
"""
from __future__ import annotations

import numpy as np

from ..constants import MSUN_SI, MEVFM3_TO_PA, NUCLEON_KG, FM3_TO_M3
from ..eos.base import EOSModel
from ..tov.family import build_family, default_hc_grid


def _to_msun(m):
    m = np.asarray(m, dtype=float)
    return np.where(m > 1e10, m / MSUN_SI, m)


class EOSConcreteNMB:
    """Duck-typed RIFT EOSConcrete backed by nmbackend's JAX TOV family."""

    def __init__(self, model: EOSModel, params: dict = None, name: str = None,
                 n_models: int = 120, n_steps: int = 800):
        self.name = name or getattr(model, "name", "nmbackend-EOS")
        self.model = model
        self.params = dict(params or {})

        table = model.eos_table(self.params)
        self.eos = table          # RIFT duck-typing: the EOS object
        self.eos_fam = None       # no lalsuite family; sequence cached below

        fam = build_family(table, default_hc_grid(table, n_models=n_models),
                           n_steps=n_steps)

        # cache stable rising branch (numpy) for fast lookup
        M = np.asarray(fam.M_msun)
        R = np.asarray(fam.R_km)
        L = np.asarray(fam.Lambda)
        stable = np.asarray(fam.stable)
        idx_max = int(np.argmax(np.where(stable, M, -np.inf)))
        sl = slice(0, idx_max + 1)
        self._M = M[sl]
        self._R = R[sl]
        self._L = L[sl]
        self.mMaxMsun = float(M[idx_max])

        # EOS table in SI for pressure_density_on_grid
        n = np.asarray(table.n)                      # fm^-3
        self._rho_si = n * NUCLEON_KG / FM3_TO_M3    # kg/m^3 (rest-mass density)
        self._p_si = np.asarray(table.p_mevfm3) * MEVFM3_TO_PA   # Pa

    # ---- RIFT EOSConcrete surface ----
    def lambda_from_m(self, m):
        m_msun = float(_to_msun(np.atleast_1d(m))[0])
        if m_msun >= self.mMaxMsun:
            return 0.0
        return float(np.interp(m_msun, self._M, self._L))

    def lambda_from_m_vector(self, m):
        m_msun = _to_msun(m)
        out = np.interp(m_msun, self._M, self._L,
                        left=self._L[0], right=0.0)
        out = np.where(m_msun >= self.mMaxMsun, 0.0, out)
        return out

    def radius_from_m(self, m):
        m_msun = float(_to_msun(np.atleast_1d(m))[0])
        return float(np.interp(m_msun, self._M, self._R))

    def pressure_density_on_grid(self, logrho_grid, enforce_causal=False):
        """log10 p[SI, Pa] on a grid of log10 rho[SI, kg/m^3] (rest-mass density)."""
        logrho_tab = np.log10(self._rho_si)
        logp_tab = np.log10(self._p_si)
        return np.interp(np.asarray(logrho_grid), logrho_tab, logp_tab)

    def mass_radius_lambda(self):
        """Return cached stable-branch (M[Msun], R[km], Lambda) arrays."""
        return self._M.copy(), self._R.copy(), self._L.copy()

    def __repr__(self):
        return (f"<EOSConcreteNMB {self.name} "
                f"mMax={self.mMaxMsun:.3f} Msun, params={self.params}>")
