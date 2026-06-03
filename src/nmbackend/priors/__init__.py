"""Nonparametric EOS priors.

Currently: the constrained Brownian-bridge prior of Gorda, Komoltsev, Kurkela &
Sunde (arXiv:2512.18044) -- stable, causal, thermodynamically consistent EOS draws
by construction, bridging a low-density nuclear anchor (our metamodel band) to the
perturbative-QCD limit. See :mod:`nmbackend.priors.bridge`.
"""
from .bridge import (kk_volume, sample_point_in_volume, fractal_refine,
                     diffuse_mu_of_n, draw_bridge_eos, draw_prior_ensemble,
                     pqcd_anchor, PQCD_TABLE)

__all__ = ["kk_volume", "sample_point_in_volume", "fractal_refine",
           "diffuse_mu_of_n", "draw_bridge_eos", "draw_prior_ensemble",
           "pqcd_anchor", "PQCD_TABLE"]
