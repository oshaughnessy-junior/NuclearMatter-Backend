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
# NOTE on other parameter sets (e.g. NL3): a naive conversion of the published
# (g_sigma, g2, g3) to this (g/m)^2 + b,c convention did NOT reproduce NL3's
# saturation -- the scalar gap equation had NO root at NL3's known physical point
# (the meson terms came out ~2x too small vs rho_s), i.e. a coupling-normalization
# mismatch, not a solver problem (GM1 is exact through the same code).  Sets with
# c < 0 also have a nonlinear potential unbounded below (metastable physical branch).
# Adding a new set therefore requires verifying its couplings against its own
# saturation point first.  See docs/backlog.md.
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


def _solve_phi(n_n, n_p, p, ngrid=320, nbis=48):
    """Self-consistent scalar field Phi (fm^-1) by FIRST-ROOT bracketed bisection.

    The scalar gap equation can be non-convex (sets with b, c < 0, e.g. NL3) with
    several roots.  The *physical* branch is the one continuously connected from
    m* = M (Phi = 0) at zero density, which is always the **lowest-Phi root** (resid
    starts negative at Phi=0 and the first up-crossing is physical; higher crossings
    are spurious).  We locate the first '-' -> '+' sign change of resid on a Phi grid
    and bisect there -- robust for ALL parameter sets and fully vectorized over the
    density array.  Differentiable (the bracket index is piecewise-constant, like a
    searchsorted; gradients flow through the bisected endpoints).
    """
    M = p["M_mev"] / HBARC
    Cs, b, c = p["C_sigma"], p["b"], p["c"]
    kfn, kfp = _kf(n_n), _kf(n_p)
    shape = jnp.shape(kfn)
    kfn = jnp.reshape(kfn, (-1,))
    kfp = jnp.reshape(kfp, (-1,))                    # (D,)

    def resid(Phi):                                  # broadcasts Phi with (D,) kf
        mstar = M - Phi
        rho_s = _rho_s_one(kfn, mstar) + _rho_s_one(kfp, mstar)
        return Phi / Cs + b * M * Phi**2 + c * Phi**3 - rho_s

    Phi_grid = jnp.linspace(1e-6, 0.985 * M, ngrid)  # (G,)
    rg = resid(Phi_grid[:, None])                    # (G, D)
    cross = (rg[:-1] < 0.0) & (rg[1:] >= 0.0)        # first up-crossing per column
    i = jnp.argmax(cross, axis=0)                    # (D,)
    lo, hi = Phi_grid[i], Phi_grid[i + 1]            # (D,)

    def body(_k, c2):
        lo, hi = c2
        mid = 0.5 * (lo + hi)
        r = resid(mid)
        return jnp.where(r < 0.0, mid, lo), jnp.where(r < 0.0, hi, mid)

    lo, hi = jax.lax.fori_loop(0, nbis, body, (lo, hi))
    return jnp.reshape(0.5 * (lo + hi), shape)


def _leptons(n_charge):
    """Solve the lepton sector at fixed charge density n_charge = n_p (fm^-3).

    Electrons + muons in chemical equilibrium (common mu_e = mu_mu) with charge
    neutrality n_e + n_mu = n_charge.  Muons switch on once mu_e > m_mu (~106 MeV).
    Returns (mu_e [fm^-1], n_e, n_mu [fm^-3], eps_lep [fm^-4]).
    """
    m_e, m_mu = M_E / HBARC, M_MU / HBARC
    n_charge = jnp.maximum(n_charge, 1e-30)

    def nlep(mu):
        ke = jnp.sqrt(jnp.maximum(mu**2 - m_e**2, 0.0))
        kmu = jnp.sqrt(jnp.maximum(mu**2 - m_mu**2, 0.0))
        return (ke**3 + kmu**3) / (3.0 * _PI2)

    mu_lo = jnp.full_like(n_charge, m_e)
    mu_hi = jnp.sqrt((3.0 * _PI2 * n_charge) ** (2.0 / 3.0) + m_e**2) * 1.0001

    def body(_i, c):
        lo, hi = c
        mid = 0.5 * (lo + hi)
        f = nlep(mid) - n_charge           # increasing in mu
        return jnp.where(f < 0.0, mid, lo), jnp.where(f < 0.0, hi, mid)

    lo, hi = jax.lax.fori_loop(0, 50, body, (mu_lo, mu_hi))
    mu_e = 0.5 * (lo + hi)
    ke = jnp.sqrt(jnp.maximum(mu_e**2 - m_e**2, 0.0))
    kmu = jnp.sqrt(jnp.maximum(mu_e**2 - m_mu**2, 0.0))
    n_e = ke**3 / (3.0 * _PI2)
    n_mu = kmu**3 / (3.0 * _PI2)
    eps_lep = _eps_kin_one(ke, m_e) + _eps_kin_one(kmu, m_mu)   # =0 if kf=0
    return mu_e, n_e, n_mu, eps_lep


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
        mu_e_actual = _leptons(x_p * n_B)[0]        # e + mu, charge neutrality
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
    """eps, p, mu_n at (n_B, x_p) with Phi solved per-density (independent).

    Used for the per-density paths and as a back-compatible wrapper.  The EOS table
    instead uses the continuation solver (``_solve_grid``) for robustness with
    strongly nonlinear sets.  ``leptons=False`` -> pure nuclear matter (no leptons).
    """
    n_p, n_n = x_p * n_B, (1.0 - x_p) * n_B
    Phi = _solve_phi(n_n, n_p, p)
    return _eps_p_from_state(n_B, x_p, Phi, p, leptons)


def _eps_p_from_state(n_B, x_p, Phi, p, leptons=True):
    """eps (MeV/fm^3), p (MeV/fm^3), mu_n (MeV) given a *solved* scalar field Phi."""
    M = p["M_mev"] / HBARC
    Cs, Co, Cr, b, c = p["C_sigma"], p["C_omega"], p["C_rho"], p["b"], p["c"]
    n_p, n_n = x_p * n_B, (1.0 - x_p) * n_B
    mstar = M - Phi
    kfn, kfp = _kf(n_n), _kf(n_p)
    n3 = n_p - n_n
    W = Co * n_B
    R = 0.5 * Cr * n3

    eps_kin = _eps_kin_one(kfn, mstar) + _eps_kin_one(kfp, mstar)
    eps_field = (0.5 * Phi**2 / Cs + (1.0 / 3.0) * b * M * Phi**3
                 + 0.25 * c * Phi**4 + 0.5 * Co * n_B**2 + 0.5 * R**2 / Cr)
    eps_lep = _leptons(x_p * n_B)[3] if leptons else 0.0 * n_B  # e + mu (n_p charge)
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


def symmetry_energy(n_B, Phi, p):
    """Symmetry energy S(n) [MeV] (analytic RMF form) given the symmetric-matter
    scalar field Phi: S = (1/6) k_F^2/E*_F + (1/8) C_rho n."""
    mstar = p["M_mev"] / HBARC - Phi
    kf = _kf(n_B / 2.0)
    ef = jnp.sqrt(kf**2 + mstar**2)
    return ((1.0 / 6.0) * kf**2 / ef + (1.0 / 8.0) * p["C_rho"] * n_B) * HBARC


def saturation_properties(params=None, n_start=0.005, n_hi=0.22, npts=440):
    """Return GM1/NL3-style saturation diagnostics for a parameter set:
    n0, B/A (MeV), K (MeV), m*/M, E_sym (MeV), L (MeV).

    The scan starts at LOW density (m* ~ M, where the scalar root is unambiguous)
    so the continuation solver tracks the physical branch up to saturation -- this
    is what makes it work for strongly nonlinear sets (NL3) and not just GM1."""
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    import numpy as np
    M = p["M_mev"]
    ns = jnp.linspace(n_start, n_hi, npts)
    Phi_arr = _solve_phi(ns / 2.0, ns / 2.0, p)        # symmetric, robust first-root
    eps, pres, _ = jax.vmap(lambda n, Ph: _eps_p_from_state(n, 0.5, Ph, p,
                                                            leptons=False))(ns, Phi_arr)
    Esym_arr = jax.vmap(lambda n, Ph: symmetry_energy(n, Ph, p))(ns, Phi_arr)
    eps, pres, ns_np = np.asarray(eps), np.asarray(pres), np.asarray(ns)
    Phi_np, Esym_np = np.asarray(Phi_arr), np.asarray(Esym_arr)
    # saturation = binding-energy minimum (where p = 0 with dp/dn > 0), NOT the
    # trivial p ~ 0 at the low-density boundary.
    i0 = int(np.argmin(eps / ns_np - M))
    n0 = float(ns_np[i0])
    BA = float(eps[i0] / n0 - M)
    K = float(9.0 * (pres[i0 + 1] - pres[i0 - 1]) / (ns_np[i0 + 1] - ns_np[i0 - 1]))
    mstar_frac = (M / HBARC - float(Phi_np[i0])) * HBARC / M
    Esym = float(Esym_np[i0])
    L = float(3.0 * n0 * (Esym_np[i0 + 1] - Esym_np[i0 - 1])
              / (ns_np[i0 + 1] - ns_np[i0 - 1]))
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
        x_p = _solve_xp(n, p)                              # beta-equilibrium
        Phi = _solve_phi((1.0 - x_p) * n, x_p * n, p)      # robust first-root, vec
        eps, pres, _ = jax.vmap(lambda ni, xi, Ph: _eps_p_from_state(ni, xi, Ph, p))(
            n, x_p, Phi)
        pres = jax.lax.cummax(jnp.maximum(pres, 1e-9))
        if self.n_surface is not None:
            n, eps, pres = extend_to_surface(n, eps, pres,
                                             n_surface=self.n_surface,
                                             gamma_crust=self.gamma_crust)
        return build_h_and_cs2(n, eps, pres)
