"""Nuclear empirical-parameter "metamodel" EOS (physical, parameter-based).

A cold, charge-neutral, beta-equilibrium EOS of nucleons + leptons built from the
standard nuclear empirical parameters (NEPs).  The isoscalar and isovector energy
per nucleon are Taylor-expanded around saturation:

    x(n)   = (n - n_sat) / (3 n_sat)
    e_sat  = E_sat + 1/2 K_sat x^2 + 1/6 Q_sat x^3
    S(n)   = E_sym + L_sym x + 1/2 K_sym x^2 + 1/6 Q_sym x^3
    e_nuc  = e_sat(n) + S(n) * delta^2,     delta = 1 - 2 x_p   (parabolic approx)

Leptons are relativistic Fermi gases with charge neutrality x_p = x_e + x_mu.
beta-equilibrium fixes the proton fraction x_p(n) via mu_e = (M_N - M_P) + 4 S delta.
By default only electrons are included (``MetaModelEOS(muons=False)``, the
historical behaviour, kept so existing tables are unchanged).  Pass
``muons=True`` whenever the composition matters: muons lower the direct-Urca
threshold proton fraction below the npe value 1/9 and change x_p above
mu_e ~ m_mu (n ~ n_sat).
Pressure follows from the *envelope theorem*: because total energy density is
stationary in x_p at equilibrium, p = n d(eps)/dn - eps can be taken at fixed x_p.

A bare Taylor expansion is only trustworthy near saturation and softens
unphysically when extrapolated to several n_sat.  Following the standard practice
(MetaModel+CSE in ``jester``; constant-speed-of-sound extensions), above a matching
density ``n_match`` (default 2 n_sat) we append a **constant-sound-speed (CSS)**
causal segment with dp/deps = cs2_tail.  ``cs2_tail`` thus becomes a physical
stiffness parameter controlling the maximum mass -- a natural inference knob.

This is the empirical-parameter ("physical") nuclear-matter model emphasized in the
project notes.  A self-consistent RMFT Lagrangian solver and a proper crust
(SLy/BPS) are planned siblings under the same EOSModel interface.

Motivated by: Margueron, Hoffmann Casali & Gulminelli PRC 97 (2018) 025805; the
``jester`` MetaModel (MIT); the RMF parameterization in Z. Zhu's ``fisher_eos``
(motivated-by only -- see ATTRIBUTION.md).
"""
from __future__ import annotations

import math
import jax
import jax.numpy as jnp

from ..constants import HBARC, M_N, M_P, M_E, N_SAT_DEFAULT
from .base import EOSModel, build_h_and_cs2, extend_to_surface, Composition
from .leptons import lepton_sector

jax.config.update("jax_enable_x64", True)

#: Default NEPs (MeV; n_sat in fm^-3) plus high-density extension controls.
DEFAULT_PARAMS = dict(
    n_sat=N_SAT_DEFAULT,
    E_sat=-16.0, K_sat=230.0, Q_sat=-300.0,
    E_sym=32.0, L_sym=60.0, K_sym=-100.0, Q_sym=0.0,
    cs2_tail=0.4,        # constant sound speed^2 of the causal high-density segment
    n_match_fac=2.0,     # matching density in units of n_sat
)

_PI = math.pi


def _eps_electron(n_e):
    """Relativistic electron-gas energy density, MeV/fm^3 (n_e in fm^-3)."""
    n_e = jnp.maximum(n_e, 1e-30)
    kF = HBARC * (3.0 * _PI**2 * n_e) ** (1.0 / 3.0)        # MeV
    m = M_E
    EF = jnp.sqrt(kF**2 + m**2)
    term = kF * EF * (2.0 * kF**2 + m**2) - m**4 * jnp.log((kF + EF) / m)
    return term / (8.0 * _PI**2 * HBARC**3)


def _mu_electron(n_e):
    """Electron chemical potential mu_e = E_F, MeV."""
    n_e = jnp.maximum(n_e, 1e-30)
    kF = HBARC * (3.0 * _PI**2 * n_e) ** (1.0 / 3.0)
    return jnp.sqrt(kF**2 + M_E**2)


def _esat_S(n, p):
    """Isoscalar e_sat(n) and symmetry energy S(n), MeV."""
    x = (n - p["n_sat"]) / (3.0 * p["n_sat"])
    e_sat = p["E_sat"] + 0.5 * p["K_sat"] * x**2 + (1.0 / 6.0) * p["Q_sat"] * x**3
    S = (p["E_sym"] + p["L_sym"] * x + 0.5 * p["K_sym"] * x**2
         + (1.0 / 6.0) * p["Q_sym"] * x**3)
    return e_sat, S


def _eps_total(n, x_p, p, muons=False):
    """Total energy density (nucleons + leptons), MeV/fm^3, incl. rest mass."""
    e_sat, S = _esat_S(n, p)
    delta = 1.0 - 2.0 * x_p
    e_per_baryon = (x_p * M_P + (1.0 - x_p) * M_N + e_sat + S * delta**2)
    eps_b = n * e_per_baryon
    if muons:
        eps_l = lepton_sector(x_p * n, muons=True)[3] * HBARC
    else:
        eps_l = _eps_electron(x_p * n)
    return eps_b + eps_l


def _solve_xp(n, p, n_iter=70, muons=False):
    """beta-equilibrium proton fraction x_p(n), differentiable w.r.t. params.

    The forward solve is a robust vectorized bisection (non-differentiable); we
    then take a single Newton step at the converged root so that AD recovers the
    correct implicit derivative dx_p/dparam = -(dg/dparam)/(dg/dx_p).
    """
    _, S = _esat_S(n, p)

    def g(x_p):
        delta = 1.0 - 2.0 * x_p
        if muons:       # charge neutrality n_p = n_e + n_mu, mu_mu = mu_e
            mu_e = lepton_sector(x_p * n, muons=True)[0] * HBARC
        else:
            mu_e = _mu_electron(x_p * n)
        return mu_e - ((M_N - M_P) + 4.0 * S * delta)

    lo = jnp.full_like(n, 1e-6)
    hi = jnp.full_like(n, 0.5 - 1e-9)

    def body(_, carry):
        lo, hi = carry
        mid = 0.5 * (lo + hi)
        gm = g(mid)            # g increasing in x_p
        lo = jnp.where(gm < 0.0, mid, lo)
        hi = jnp.where(gm < 0.0, hi, mid)
        return lo, hi

    lo, hi = jax.lax.fori_loop(0, n_iter, body, (lo, hi))
    x0 = jax.lax.stop_gradient(0.5 * (lo + hi))

    # one differentiable Newton step (implicit differentiation)
    gx = g(x0)
    dg = jax.grad(lambda xp: jnp.sum(g(xp)))(x0)
    return x0 - gx / dg


class MetaModelEOS(EOSModel):
    """Empirical-parameter nuclear-matter EOS (NEP metamodel) + CSS tail."""

    name = "MetaModelEOS"

    def __init__(self, n_min=3e-4, n_points_low=240, n_points_tail=160,
                 eps_max_fac=14.0, n_surface=1e-7, gamma_crust=1.3,
                 muons=False):
        self.muons = muons
        self.n_min = n_min
        self.n_points_low = n_points_low
        self.n_points_tail = n_points_tail
        self.eps_max_fac = eps_max_fac
        # surface continuation (see eos.base.extend_to_surface); set n_surface=None
        # to disable and recover the old truncated-at-n_min behaviour.
        self.n_surface = n_surface
        self.gamma_crust = gamma_crust

    def _low_density(self, p):
        """Taylor metamodel part on n in [n_min, n_match]; MeV/fm^3, fm^-3."""
        n_match = p["n_match_fac"] * p["n_sat"]
        n = jnp.exp(jnp.linspace(math.log(self.n_min), math.log(n_match),
                                 self.n_points_low))
        # x_p is differentiable w.r.t. params; the inner pressure derivative below
        # differentiates only w.r.t. n, so x_p is correctly held fixed there.
        mu = self.muons
        x_p = _solve_xp(n, p, muons=mu)
        eps = jax.vmap(lambda ni, xi: _eps_total(ni, xi, p, mu))(n, x_p)
        deps_dn = jax.vmap(
            lambda ni, xi: jax.grad(lambda nn: _eps_total(nn, xi, p, mu))(ni)
        )(n, x_p)
        pres = n * deps_dn - eps
        pres = jax.lax.cummax(jnp.maximum(pres, 1e-9))
        return n, eps, pres

    def _css_tail(self, n_m, eps_m, p_m, p):
        """Constant-sound-speed causal extension above the match point.

        dp/deps = cs2_tail  =>  p = p_m + cs2 (eps - eps_m).  The baryon density
        follows from dln n = deps/(eps+p), which integrates in closed form.
        """
        cs2 = p["cs2_tail"]
        eps = jnp.exp(jnp.linspace(jnp.log(eps_m * (1.0 + 1e-3)),
                                   jnp.log(eps_m * self.eps_max_fac),
                                   self.n_points_tail))
        pres = p_m + cs2 * (eps - eps_m)
        c0 = p_m - cs2 * eps_m
        lin = (1.0 + cs2) * eps + c0
        lin_m = (1.0 + cs2) * eps_m + c0
        n = n_m * (lin / lin_m) ** (1.0 / (1.0 + cs2))
        return n, eps, pres

    def composition(self, params: dict = None) -> Composition:
        """beta-equilibrium (x_p, x_e, x_mu) on the table's density grid.

        Below n_match this is the composition of the EOS itself.  Above n_match
        the pressure comes from the CSS tail, which carries no composition; there
        x_p is the beta-equilibrium solution of the Taylor S(n) extrapolated past
        n_match.  That is a modelling choice (as in MetaModel+CSE dUrca studies),
        not a property of the tail.  Leptons follow ``self.muons``.
        """
        p = dict(DEFAULT_PARAMS)
        if params:
            p.update(params)
        n_lo, eps_lo, p_lo = self._low_density(p)
        n_t, _, _ = self._css_tail(n_lo[-1], eps_lo[-1], p_lo[-1], p)
        n = jnp.concatenate([n_lo, n_t])
        x_p = jnp.clip(_solve_xp(n, p, muons=self.muons), 0.0, 0.5)
        if self.muons:
            _, n_e, n_mu, _ = lepton_sector(x_p * n, muons=True)
            x_e, x_mu = n_e / n, n_mu / n
        else:
            x_e, x_mu = x_p, jnp.zeros_like(x_p)
        return Composition(n=n, x_p=x_p, x_e=x_e, x_mu=x_mu)

    def eos_table(self, params: dict = None):
        p = dict(DEFAULT_PARAMS)
        if params:
            p.update(params)

        n_lo, eps_lo, p_lo = self._low_density(p)
        n_t, eps_t, p_t = self._css_tail(n_lo[-1], eps_lo[-1], p_lo[-1], p)

        n = jnp.concatenate([n_lo, n_t])
        eps = jnp.concatenate([eps_lo, eps_t])
        pres = jnp.concatenate([p_lo, p_t])
        pres = jax.lax.cummax(jnp.maximum(pres, 1e-9))

        # surface convention: continue to p -> 0 so R is integrated to the surface
        if self.n_surface is not None:
            n, eps, pres = extend_to_surface(n, eps, pres,
                                             n_surface=self.n_surface,
                                             gamma_crust=self.gamma_crust)
        return build_h_and_cs2(n, eps, pres)
