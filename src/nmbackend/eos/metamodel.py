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

Effective masses (:func:`landau_masses`, used only for the cooling composition).
The Taylor energy has no kinetic/potential split, so the metamodel carries no
effective mass of its own.  We take the form of Margueron, Hoffmann Casali &
Gulminelli (2018), Eqs. 15-16:

    m / m*_q = 1 + (kappa_sat + tau_q kappa_sym delta) n / n_sat,   tau_n = +1

with kappa_sat = m/m*_sat - 1 and kappa_sym fixed by the neutron-proton
splitting m*_n/m - m*_p/m = dmstar_sat in neutron matter at n_sat.  Defaults
are their central values (Sec. II A), m*_sat/m = 0.75 and Delta m*_sat/m = 0.10
(``MSTAR_DEFAULTS``; override through ``params``).  The same form is used
above n_match, like the Taylor x_p there.  The EOS itself does not depend on
these two parameters.

This is the empirical-parameter ("physical") nuclear-matter model emphasized in the
project notes.  Below n_t the table uses the SLy crust (``eos.crust``); the
legacy polytrope surface continuation is ``crust="polytrope"``.

Motivated by: Margueron, Hoffmann Casali & Gulminelli PRC 97 (2018) 025805; the
``jester`` MetaModel (MIT); the RMF parameterization in Z. Zhu's ``fisher_eos``
(motivated-by only -- see ATTRIBUTION.md).
"""
from __future__ import annotations

import math
import jax
import jax.numpy as jnp

from ..constants import HBARC, M_N, M_P, M_E, N_SAT_DEFAULT
from .base import EOSModel, Composition, compiled_table
from .leptons import lepton_sector
from .crust import (N_T_DEFAULT, check_crust, finish_table_flags, join_density,
                    monotone_core)

jax.config.update("jax_enable_x64", True)

#: Default NEPs (MeV; n_sat in fm^-3) plus high-density extension controls.
DEFAULT_PARAMS = dict(
    n_sat=N_SAT_DEFAULT,
    E_sat=-16.0, K_sat=230.0, Q_sat=-300.0,
    E_sym=32.0, L_sym=60.0, K_sym=-100.0, Q_sym=0.0,
    cs2_tail=0.4,        # constant sound speed^2 of the causal high-density segment
    n_match_fac=2.0,     # matching density in units of n_sat
)

#: Effective-mass prescription (Margueron+2018 central values); composition only.
MSTAR_DEFAULTS = dict(mstar_sat=0.75, dmstar_sat=0.10)

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


def _solve_xp(n, p, n_iter=70, muons=False, S=None):
    """beta-equilibrium proton fraction x_p(n), differentiable w.r.t. params.

    The forward solve is a robust vectorized bisection (non-differentiable); we
    then take a single Newton step at the converged root so that AD recovers the
    correct implicit derivative dx_p/dparam = -(dg/dparam)/(dg/dx_p).
    ``S`` overrides the Taylor symmetry energy on ``n`` if given.
    """
    if S is None:
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


def landau_masses(params, n, x_p):
    """(m*_n/m, m*_p/m) of the Margueron+2018 prescription (module docstring)."""
    p = dict(DEFAULT_PARAMS, **MSTAR_DEFAULTS)
    p.update(params or {})
    n = jnp.asarray(n, float)
    delta = 1.0 - 2.0 * jnp.asarray(x_p, float)
    a = 1.0 / p["mstar_sat"]                     # 1 + kappa_sat
    d = p["dmstar_sat"]
    # 1/(a + k) - 1/(a - k) = d, the root with |k| < a
    k_sym = -d * a**2 / (1.0 + jnp.sqrt(1.0 + (d * a) ** 2))
    u = n / p["n_sat"]
    inv = lambda tau: 1.0 + (a - 1.0 + tau * k_sym * delta) * u
    return 1.0 / inv(1.0), 1.0 / inv(-1.0)


class MetaModelEOS(EOSModel):
    """Empirical-parameter nuclear-matter EOS (NEP metamodel) + CSS tail."""

    name = "MetaModelEOS"

    def __init__(self, n_min=3e-4, n_points_low=240, n_points_tail=160,
                 eps_max_fac=14.0, n_surface=1e-7, gamma_crust=1.3,
                 muons=False, s_high="taylor", crust="sly", n_t=N_T_DEFAULT,
                 allow_unstable_core=False):
        if s_high not in ("taylor", "power", "power_matched"):
            raise ValueError(f"unknown s_high={s_high!r}")
        self.muons = muons
        self.s_high = s_high
        self.n_min = n_min
        self.n_points_low = n_points_low
        self.n_points_tail = n_points_tail
        self.eps_max_fac = eps_max_fac
        # low-density completion (eos.crust.finish_table): the SLy crust below
        # n_t, or with crust="polytrope" the legacy surface continuation
        # (n_surface, gamma_crust).  n_surface=None disables both and recovers
        # the old truncated-at-n_min behaviour.
        self.n_surface = n_surface
        self.gamma_crust = gamma_crust
        self.crust = check_crust(crust, n_surface)
        self.n_t = n_t
        # True keeps cores whose raw pressure falls with density (plateau
        # tables; see eos.crust.monotone_core) instead of rejecting them
        self.allow_unstable_core = allow_unstable_core

    def _low_density(self, p, n_min=None, onset=False):
        """Taylor metamodel part on n in [n_min, n_match]; MeV/fm^3, fm^-3.
        ``onset=True`` also returns ln n_u of :func:`eos.crust.monotone_core`."""
        n_match = p["n_match_fac"] * p["n_sat"]
        n_min = self.n_min if n_min is None else n_min
        n = jnp.exp(jnp.linspace(math.log(n_min), jnp.log(n_match),
                                 self.n_points_low))
        # x_p is differentiable w.r.t. params; the inner pressure derivative below
        # differentiates only w.r.t. n, so x_p is correctly held fixed there.
        mu = self.muons
        x_p = _solve_xp(n, p, muons=mu)
        eps = jax.vmap(lambda ni, xi: _eps_total(ni, xi, p, mu))(n, x_p)
        deps_dn = jax.vmap(
            lambda ni, xi: jax.grad(lambda nn: _eps_total(nn, xi, p, mu))(ni)
        )(n, x_p)
        pres, ln_nu = monotone_core(n, n * deps_dn - eps, self.n_t)
        return (n, eps, pres, ln_nu) if onset else (n, eps, pres)

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

        ``s_high`` (constructor) replaces the Taylor S(n) above n_match by a
        bounded power law S(n_match) (n/n_match)^g.  For "power",
        g = params["gamma_S"] (default 0.5); for "power_matched", g is the
        slope-matched n S'/S at n_match, clipped to [0, 1].  The pressure is
        unchanged (it comes from the tail).
        """
        p = dict(DEFAULT_PARAMS)
        if params:
            p.update(params)
        n_lo, eps_lo, p_lo = self._low_density(p)
        n_t, _, _ = self._css_tail(n_lo[-1], eps_lo[-1], p_lo[-1], p)
        n = jnp.concatenate([n_lo, n_t])
        S = None
        if self.s_high != "taylor":
            n_m = n_lo[-1]
            _, S = _esat_S(n, p)
            S_m = _esat_S(n_m, p)[1]
            if self.s_high == "power":
                g = p.get("gamma_S", 0.5)
            else:
                dS = jax.grad(lambda x: _esat_S(x, p)[1])(n_m)
                g = jnp.clip(n_m * dS / S_m, 0.0, 1.0)
            S = jnp.where(n <= n_m, S, S_m * (n / n_m) ** g)
        x_p = jnp.clip(_solve_xp(n, p, muons=self.muons, S=S), 0.0, 0.5)
        if self.muons:
            _, n_e, n_mu, _ = lepton_sector(x_p * n, muons=True)
            x_e, x_mu = n_e / n, n_mu / n
        else:
            x_e, x_mu = x_p, jnp.zeros_like(x_p)
        return Composition(n=n, x_p=x_p, x_e=x_e, x_mu=x_mu)

    def landau_masses(self, params, comp):
        """(m*_n/m, m*_p/m) on ``comp`` (see :func:`landau_masses`)."""
        return landau_masses(params, comp.n, comp.x_p)

    def eos_table(self, params: dict = None):
        p = dict(DEFAULT_PARAMS)
        if params:
            p.update(params)
        return compiled_table(self, p)

    def _table_flags(self, p):
        sly = self.crust == "sly" and self.n_surface is not None
        n_lo, eps_lo, p_lo, ln_nu = self._low_density(
            p, n_min=join_density(self.n_t) if sly else None, onset=True)
        n_t, eps_t, p_t = self._css_tail(n_lo[-1], eps_lo[-1], p_lo[-1], p)

        n = jnp.concatenate([n_lo, n_t])
        eps = jnp.concatenate([eps_lo, eps_t])
        pres = jnp.concatenate([p_lo, p_t])
        pres = jax.lax.cummax(jnp.maximum(pres, 1e-9))

        return finish_table_flags(n, eps, pres, self.crust, self.n_t,
                                  self.n_surface, self.gamma_crust, ln_nu,
                                  self.allow_unstable_core)
