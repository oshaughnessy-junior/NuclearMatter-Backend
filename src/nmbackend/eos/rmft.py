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
#: NL3 (Lalazissis, Koenig & Ring 1997).  Converted to (g/m)^2 and the
#: U = 1/3 b M (g_s s)^3 + 1/4 c (g_s s)^4 convention: b = g2/(M g_sigma^3),
#: c = g3/g_sigma^4 with g_sigma=10.217, g2=-10.431 fm^-1, g3=-28.885.
#: NOTE: b, c < 0 make the scalar gap equation NON-CONVEX (resid < 0 at both ends
#: of the m* bracket); the physical root must be tracked by CONTINUATION from low
#: density.  The current per-density damped Newton does not do this, so NL3 is not
#: yet usable.  See docs/backlog.md ("RMF continuation solver").
NL3 = dict(
    M_mev=939.0,
    C_sigma=15.739, C_omega=10.530, C_rho=1.3398,  # fm^2
    b=-0.002058, c=-0.002651,
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

    dresid = jax.grad(lambda x: jnp.sum(resid(x)))
    # damped + clamped Newton; robust for well-behaved (convex) self-couplings such
    # as GM1.  Strongly nonlinear sets (e.g. NL3, b,c < 0) have a non-convex gap
    # equation that needs a continuation solver -- see docs/backlog.md.
    def step(_i, Phi):
        return jnp.clip(Phi - 0.7 * resid(Phi) / dresid(Phi), 0.0, 0.985 * M)
    Phi0 = 0.3 * M * jnp.ones_like(kfn)            # match the density array shape
    Phi = jax.lax.fori_loop(0, 45, step, Phi0)
    Phi0 = jax.lax.stop_gradient(Phi)
    return jnp.clip(Phi0 - resid(Phi0) / dresid(Phi0), 0.0, 0.985 * M)


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


def symmetry_energy(n_B, p):
    """Symmetry energy S(n) [MeV] (analytic RMF form): the isovector curvature of
    the energy per baryon.  S = (1/6) k_F^2/E*_F + (1/8) C_rho n, with symmetric
    matter k_F and m*."""
    M = p["M_mev"] / HBARC
    Phi = _solve_phi(n_B / 2.0, n_B / 2.0, p)
    mstar = M - Phi
    kf = _kf(n_B / 2.0)
    ef = jnp.sqrt(kf**2 + mstar**2)
    return ((1.0 / 6.0) * kf**2 / ef + (1.0 / 8.0) * p["C_rho"] * n_B) * HBARC


def saturation_properties(params=None, n_lo=0.13, n_hi=0.18, npts=121):
    """Return GM1/NL3-style saturation diagnostics for a parameter set:
    n0, B/A (MeV), K (MeV), m*/M, E_sym (MeV), L (MeV)."""
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    M = p["M_mev"]
    ns = jnp.linspace(n_lo, n_hi, npts)
    sym = jax.jit(lambda n: _eps_p_of_n(n, 0.5, p, leptons=False)[:2])
    eps, pres = jax.vmap(sym)(ns)
    eps, pres, ns = map(__import__("numpy").asarray, (eps, pres, ns))
    i0 = int(abs(pres).argmin())
    n0 = float(ns[i0])
    BA = float(eps[i0] / n0 - M)
    K = float(9.0 * (pres[i0 + 1] - pres[i0 - 1]) / (ns[i0 + 1] - ns[i0 - 1]))
    Phi = float(_solve_phi(jnp.asarray(n0 / 2), jnp.asarray(n0 / 2), p))
    mstar_frac = (M / HBARC - Phi) * HBARC / M
    Esym = float(symmetry_energy(jnp.asarray(n0), p))
    L = float(3.0 * n0 * jax.grad(lambda n: symmetry_energy(n, p))(n0))
    return dict(n0=n0, BA=BA, K=K, mstar_frac=mstar_frac, E_sym=Esym, L=L)


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
