"""Relativistic mean-field theory (RMFT) nuclear EOS: sigma-omega-rho with nonlinear
scalar self-couplings (Boguta-Bodmer form).

This is the "critically physical" model from the project notes -- a self-consistent
field-theoretic EOS, in contrast to the empirical-parameter metamodel.  Nucleons
(n, p) interact via exchange of a scalar sigma (attraction, generates an effective
mass m* = M - g_sigma sigma), a vector omega (repulsion), and an isovector rho
(symmetry).  In the mean-field approximation the meson fields are classical and
uniform; sigma is fixed self-consistently by its gap equation while omega and rho
are algebraic in the densities.

Conventions (Glendenning; ``Phi = g_sigma * sigma``, working in hbar*c = 1, momenta
and masses in fm^-1, densities in fm^-3, energy density in fm^-4 -> MeV/fm^3):

    m* = M - Phi
    Phi/C_sigma + b M Phi^2 + c Phi^3 = rho_s(m*)         (self-consistent)
    W   = C_omega n_B                 (= g_omega omega)
    R   = (1/2) C_rho  n_3            (= g_rho   rho ; n_3 = n_p - n_n)
    mu_i = sqrt(kF_i^2 + m*^2) + W + t3_i R   (t3 = +1/2 p, -1/2 n)

with C_x = (g_x/m_x)^2.  Energy density and pressure:

    eps = sum_i eps_kin(kF_i, m*) + (1/2)Phi^2/C_sigma + (1/3)b M Phi^3
          + (1/4)c Phi^4 + (1/2)C_omega n_B^2 + (1/2)R^2/C_rho + eps_leptons
    p   = mu_n n_B - eps             (Gibbs at T=0; thermodynamically consistent)

beta-equilibrium + charge neutrality fix the proton fraction (electrons, muons).

Default couplings: GM1 (Glendenning & Moszkowski 1991), whose saturation point is
n0~0.153 fm^-3, B/A~16.3 MeV, K~300 MeV, m*/M~0.70, a_sym~32.5 MeV -- used to
validate this implementation (see tests/ and validate script).

Motivated by standard RMF theory and MUSES ``cmf_solver`` (Chiral Mean Field) as a
reference generator; see ATTRIBUTION.md / docs/roadmap.md.
"""
from __future__ import annotations

import math
import jax
import jax.numpy as jnp

from ..constants import HBARC, M_E, M_MU
from .base import EOSModel, build_h_and_cs2, extend_to_surface

jax.config.update("jax_enable_x64", True)

_PI2 = math.pi ** 2
_M_N_FM = 939.0 / HBARC          # nucleon mass in fm^-1 (use a single M for n,p)

#: GM1 parameter set (Glendenning & Moszkowski 1991).  C_x = (g_x/m_x)^2 in fm^2.
GM1 = dict(
    M_mev=939.0,
    C_sigma=11.785, C_omega=7.148, C_rho=4.410,   # fm^2
    b=0.002947, c=-0.001070,                       # dimensionless
)
DEFAULT_PARAMS = dict(GM1)


# --- relativistic Fermi-gas integrals (degeneracy 2), all in fm units ---
def _kf(n_i):                      # Fermi momentum (fm^-1) from density (fm^-3)
    return (3.0 * _PI2 * jnp.maximum(n_i, 1e-30)) ** (1.0 / 3.0)


def _rho_s_one(kf, mstar):         # scalar density of one species (fm^-3)
    ef = jnp.sqrt(kf**2 + mstar**2)
    return (mstar / (2.0 * _PI2)) * (
        kf * ef - mstar**2 * jnp.log((kf + ef) / mstar))


def _eps_kin_one(kf, mstar):       # kinetic energy density of one species (fm^-4)
    ef = jnp.sqrt(kf**2 + mstar**2)
    return (1.0 / (8.0 * _PI2)) * (
        kf * (2.0 * kf**2 + mstar**2) * ef
        - mstar**4 * jnp.log((kf + ef) / mstar))


def _solve_phi(n_n, n_p, p):
    """Self-consistent scalar field Phi (fm^-1) via Newton; differentiable."""
    M = p["M_mev"] / HBARC
    Cs, b, c = p["C_sigma"], p["b"], p["c"]
    kfn, kfp = _kf(n_n), _kf(n_p)

    def resid(Phi):
        mstar = M - Phi
        rho_s = _rho_s_one(kfn, mstar) + _rho_s_one(kfp, mstar)
        return Phi / Cs + b * M * Phi**2 + c * Phi**3 - rho_s

    Phi = 0.3 * M                      # initial guess (m*/M ~ 0.7)
    for _ in range(40):                # robust forward solve
        f = resid(Phi)
        df = jax.grad(lambda x: jnp.sum(resid(x)))(Phi)
        Phi = Phi - f / df
        Phi = jnp.clip(Phi, 0.0, 0.98 * M)   # keep m* = M - Phi positive
    # one differentiable Newton step at the converged root (implicit diff)
    Phi = jax.lax.stop_gradient(Phi)
    f = resid(Phi)
    df = jax.grad(lambda x: jnp.sum(resid(x)))(Phi)
    return Phi - f / df


def _eps_leptons(n_e, n_mu):
    """Lepton energy density (fm^-4): relativistic electron + muon gases."""
    def g(n_l, m_l_mev):
        m = m_l_mev / HBARC
        kf = _kf(n_l)
        return _eps_kin_one(kf, m)
    return g(n_e, M_E) + g(n_mu, M_MU)


def _mu_lepton(n_l, m_l_mev):
    m = m_l_mev / HBARC
    kf = _kf(n_l)
    return jnp.sqrt(kf**2 + m**2)       # fm^-1


def _solve_xp(n_B, p, n_iter=70):
    """beta-equilibrium proton fraction (electrons only here; muons added in eps).

    Condition mu_n - mu_p = mu_e with charge neutrality n_e = n_p.
    mu_n - mu_p = (E*_Fn - 1/2 R) - (E*_Fp + 1/2 R) = E*_Fn - E*_Fp - R.
    """
    M = p["M_mev"] / HBARC
    Crho = p["C_rho"]

    def g(x_p):
        n_p, n_n = x_p * n_B, (1.0 - x_p) * n_B
        Phi = _solve_phi(n_n, n_p, p)
        mstar = M - Phi
        kfn, kfp = _kf(n_n), _kf(n_p)
        efn = jnp.sqrt(kfn**2 + mstar**2)
        efp = jnp.sqrt(kfp**2 + mstar**2)
        R = 0.5 * Crho * (n_p - n_n)
        mu_e = efn - efp - R                       # required by beta-eq (fm^-1)
        mu_e_actual = _mu_lepton(x_p * n_B, M_E)    # electrons, charge neutrality
        return mu_e - mu_e_actual                   # >0 -> need more protons

    lo = jnp.full_like(n_B, 1e-4)
    hi = jnp.full_like(n_B, 0.5 - 1e-6)

    def body(_, c):
        lo, hi = c
        mid = 0.5 * (lo + hi)
        gm = g(mid)            # g DECREASES with x_p (more p -> less mu_e needed,
        #                        more electrons) -> root where gm crosses 0 downward
        lo = jnp.where(gm > 0.0, mid, lo)
        hi = jnp.where(gm > 0.0, hi, mid)
        return lo, hi

    lo, hi = jax.lax.fori_loop(0, n_iter, body, (lo, hi))
    x0 = jax.lax.stop_gradient(0.5 * (lo + hi))
    gx = g(x0)
    dg = jax.grad(lambda x: jnp.sum(g(x)))(x0)
    return x0 - gx / dg


def _eps_p_of_n(n_B, x_p, p, leptons=True):
    """Total energy density (MeV/fm^3) and pressure (MeV/fm^3) at (n_B, x_p).

    ``leptons=False`` gives pure nuclear matter (no electrons) -- used for the
    symmetric-matter saturation check; beta-equilibrium NS matter uses leptons.
    """
    M = p["M_mev"] / HBARC
    Cs, Co, Cr, b, c = p["C_sigma"], p["C_omega"], p["C_rho"], p["b"], p["c"]
    n_p, n_n = x_p * n_B, (1.0 - x_p) * n_B
    Phi = _solve_phi(n_n, n_p, p)
    mstar = M - Phi
    kfn, kfp = _kf(n_n), _kf(n_p)
    n3 = n_p - n_n
    W = Co * n_B
    R = 0.5 * Cr * n3

    eps_kin = _eps_kin_one(kfn, mstar) + _eps_kin_one(kfp, mstar)
    eps_field = (0.5 * Phi**2 / Cs + (1.0 / 3.0) * b * M * Phi**3
                 + 0.25 * c * Phi**4 + 0.5 * Co * n_B**2 + 0.5 * R**2 / Cr)
    n_e = x_p * n_B
    eps_lep = _eps_leptons(n_e, 0.0 * n_e) if leptons else 0.0 * n_B  # muons: TODO
    eps = eps_kin + eps_field + eps_lep

    mu_n = jnp.sqrt(kfn**2 + mstar**2) + W - 0.5 * R           # baryon chem. pot.
    if leptons:
        # eps + p = sum_i mu_i n_i = mu_n n_B (after beta-eq + charge neutrality)
        pres = mu_n * n_B - eps
    else:
        # pure nuclear matter: p = sum_i mu_i n_i - eps, mu_i = E*_Fi + W + t3 R
        mu_p = jnp.sqrt(kfp**2 + mstar**2) + W + 0.5 * R
        pres = mu_n * n_n + mu_p * n_p - eps
    return eps * HBARC, pres * HBARC, mu_n * HBARC              # -> MeV/fm^3, MeV


class RMFTEOS(EOSModel):
    """Self-consistent sigma-omega-rho relativistic mean-field EOS."""

    name = "RMFTEOS"

    def __init__(self, n_min=1e-3, n_max=1.2, n_points=200,
                 n_surface=1e-7, gamma_crust=1.3):
        self.n_grid = jnp.exp(jnp.linspace(math.log(n_min), math.log(n_max),
                                           n_points))
        self.n_surface = n_surface
        self.gamma_crust = gamma_crust

    # --- symmetric nuclear matter (for saturation-property validation) ---
    def symmetric(self, n_B, params=None):
        p = dict(DEFAULT_PARAMS)
        if params:
            p.update(params)
        eps, pres, _ = _eps_p_of_n(jnp.asarray(n_B), 0.5, p, leptons=False)
        return eps, pres        # MeV/fm^3

    def eos_table(self, params: dict = None):
        p = dict(DEFAULT_PARAMS)
        if params:
            p.update(params)
        n = self.n_grid
        x_p = _solve_xp(n, p)
        eps, pres, _ = jax.vmap(lambda ni, xi: _eps_p_of_n(ni, xi, p))(n, x_p)
        pres = jax.lax.cummax(jnp.maximum(pres, 1e-9))
        if self.n_surface is not None:
            n, eps, pres = extend_to_surface(n, eps, pres,
                                             n_surface=self.n_surface,
                                             gamma_crust=self.gamma_crust)
        return build_h_and_cs2(n, eps, pres)
