"""Single relativistic polytrope -- a clean, analytic EOS for solver validation.

p = K n^Gamma  (MeV/fm^3),  eps = n m_b + p/(Gamma-1).  Not meant as a physical
NS EOS; it gives an independent, well-behaved barotrope to exercise and validate
the TOV solver and the AD machinery.
"""
from __future__ import annotations

import math
import jax.numpy as jnp

from ..constants import M_BARYON
from .base import EOSModel, build_h_and_cs2

DEFAULT_PARAMS = dict(K=120.0, Gamma=2.0)


class PolytropeEOS(EOSModel):
    name = "PolytropeEOS"

    def __init__(self, n_min=1e-4, n_max=2.5, n_points=400):
        self.n_grid = jnp.exp(
            jnp.linspace(math.log(n_min), math.log(n_max), n_points)
        )

    def eos_table(self, params: dict = None):
        p = dict(DEFAULT_PARAMS)
        if params:
            p.update(params)
        n = self.n_grid
        pres = p["K"] * n ** p["Gamma"]                 # MeV/fm^3
        eps = n * M_BARYON + pres / (p["Gamma"] - 1.0)  # MeV/fm^3
        return build_h_and_cs2(n, eps, pres)
