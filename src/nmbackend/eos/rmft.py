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
and masses in fm^-1, densities in fm^-3, energy density in fm^-4 -> MeV/fm^3)::

    m* = M - Phi
    Phi/C_sigma + b M Phi^2 + c Phi^3 = rho_s(m*)         (self-consistent)
    W   = C_omega n_B                 (= g_omega omega)
    R   = (1/2) C_rho  n_3            (= g_rho   rho ; n_3 = n_p - n_n)
    (with an omega-rho coupling Lambda_v, W and R solve the coupled pair
     n_B = W/C_omega + 2 Lambda_v W R^2,  n_3/2 = R/C_rho + 2 Lambda_v W^2 R)
    mu_i = sqrt(kF_i^2 + m*^2) + W + t3_i R   (t3 = +1/2 p, -1/2 n)

with C_x = (g_x/m_x)^2.  Energy density and pressure::

    eps = sum_i eps_kin(kF_i, m*) + (1/2)Phi^2/C_sigma + (1/3)b M Phi^3
          + (1/4)c Phi^4 + (1/2)W^2/C_omega + (1/2)R^2/C_rho
          + 3 Lambda_v W^2 R^2 + eps_leptons
    p   = mu_n n_B - eps             (Gibbs at T=0; thermodynamically consistent)

beta-equilibrium (mu_n - mu_p = mu_e = mu_mu) + charge neutrality
(n_p = n_e + n_mu) fix the proton fraction; see :meth:`RMFTEOS.composition`.

The optional isoscalar-isovector coupling ``Lambda_v`` (Horowitz & Piekarewicz,
PRL 86, 5647 (2001); the FSU family) softens the symmetry energy at fixed
saturation point and E_sym.  ``couplings_from_saturation`` maps (n0, B/A, K,
m*/M, E_sym, L) to couplings, so an RMF family can be sampled in saturation
properties.

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

from ..constants import HBARC
from .base import EOSModel, Composition
from .leptons import lepton_sector
from .crust import (N_T_DEFAULT, check_crust, core_grid, finish_table,
                    monotone_core)

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
    """Electrons + muons at charge density n_charge = n_p (fm^-3); see
    :func:`nmbackend.eos.leptons.lepton_sector`.
    Returns (mu_e [fm^-1], n_e, n_mu [fm^-3], eps_lep [fm^-4])."""
    return lepton_sector(n_charge, muons=True)


def _vector_fields(n_B, n3, p, n_iter=40):
    """Mean vector fields W = g_omega omega, R = g_rho rho (fm^-1).

    Without the omega-rho coupling they are algebraic.  With ``Lambda_v`` they
    solve n_B = W/C_omega + 2 Lambda_v W R^2 and n3/2 = R/C_rho + 2 Lambda_v W^2 R,
    by fixed-point iteration (a contraction: 2 Lambda_v R^2 << 1/C_omega).
    """
    Co, Cr = p["C_omega"], p["C_rho"]
    lam = p.get("Lambda_v", 0.0)
    W, R = Co * n_B, 0.5 * Cr * n3
    if isinstance(lam, (int, float)) and lam == 0.0:
        return W, R

    def body(_i, c):
        W, R = c
        W = n_B / (1.0 / Co + 2.0 * lam * R**2)
        R = 0.5 * n3 / (1.0 / Cr + 2.0 * lam * W**2)
        return W, R

    return jax.lax.fori_loop(0, n_iter, body, (W, R))


def landau_masses(params, n, x_p):
    """(m*_n/M, m*_p/M): Landau masses at the Fermi surface.

    The single-particle energy is sqrt(k^2 + m_D^2) + W + t3 R, so the Landau
    mass k_F / (d e/dk) is E*_F = sqrt(k_F^2 + m_D^2), with Dirac mass
    m_D = M - Phi (the ``mstar_frac`` of :func:`couplings_from_saturation`)."""
    p = dict(DEFAULT_PARAMS)
    p.update(params or {})
    M = p["M_mev"] / HBARC
    n = jnp.asarray(n, float)
    x_p = jnp.asarray(x_p, float)
    n_n, n_p = (1.0 - x_p) * n, x_p * n
    m_D = M - _solve_phi(n_n, n_p, p)
    ef = lambda kf: jnp.sqrt(kf**2 + m_D**2) / M
    return ef(_kf(n_n)), ef(_kf(n_p))


def _solve_xp(n_B, p, n_iter=70):
    """beta-equilibrium proton fraction with electrons AND muons.

    Condition mu_n - mu_p = mu_e = mu_mu, with charge neutrality
    n_p = n_e + n_mu (``_leptons``).
    mu_n - mu_p = (E*_Fn - 1/2 R) - (E*_Fp + 1/2 R) = E*_Fn - E*_Fp - R.
    """
    M = p["M_mev"] / HBARC

    def g(x_p):
        n_p, n_n = x_p * n_B, (1.0 - x_p) * n_B
        Phi = _solve_phi(n_n, n_p, p)
        mstar = M - Phi
        kfn, kfp = _kf(n_n), _kf(n_p)
        efn = jnp.sqrt(kfn**2 + mstar**2)
        efp = jnp.sqrt(kfp**2 + mstar**2)
        _, R = _vector_fields(n_B, n_p - n_n, p)
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
    lam = p.get("Lambda_v", 0.0)
    n_p, n_n = x_p * n_B, (1.0 - x_p) * n_B
    mstar = M - Phi
    kfn, kfp = _kf(n_n), _kf(n_p)
    W, R = _vector_fields(n_B, n_p - n_n, p)

    eps_kin = _eps_kin_one(kfn, mstar) + _eps_kin_one(kfp, mstar)
    eps_field = (0.5 * Phi**2 / Cs + (1.0 / 3.0) * b * M * Phi**3
                 + 0.25 * c * Phi**4 + 0.5 * W**2 / Co + 0.5 * R**2 / Cr
                 + 3.0 * lam * W**2 * R**2)
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
    scalar field Phi: S = (1/6) k_F^2/E*_F + (1/8) C_rho_eff n, with
    C_rho_eff = C_rho / (1 + 2 Lambda_v C_rho W^2) and W = C_omega n."""
    mstar = p["M_mev"] / HBARC - Phi
    kf = _kf(n_B / 2.0)
    ef = jnp.sqrt(kf**2 + mstar**2)
    W = p["C_omega"] * n_B
    cr_eff = p["C_rho"] / (1.0 + 2.0 * p.get("Lambda_v", 0.0) * p["C_rho"] * W**2)
    return ((1.0 / 6.0) * kf**2 / ef + (1.0 / 8.0) * cr_eff * n_B) * HBARC


def couplings_from_saturation(n0=0.153, BA=-16.3, K=300.0, mstar_frac=0.70,
                              J=32.5, L=None, M_mev=939.0):
    """RMF couplings reproducing a saturation point (Glendenning's inversion).

    Inputs in fm^-3 (n0) and MeV (BA, K, J, L).  C_omega follows from p = 0,
    (C_sigma, b, c) from the linear system {gap equation, E/A, K} at n0, and
    C_rho from J.  If ``L`` is given, an omega-rho coupling Lambda_v lowers the
    slope from its Lambda_v = 0 value L0 at fixed J; L must lie in
    (L0 - 6 J_rho, L0], where J_rho = C_rho n0/8 is the rho part of J.

    Returns a params dict (C_sigma, C_omega, C_rho, b, c, Lambda_v, M_mev) plus
    ``L0``.  Raises ValueError if the saturation point is not reachable
    (C_sigma <= 0, J below its kinetic part, or L outside the range above).
    """
    import numpy as np
    M = M_mev / HBARC
    mstar = mstar_frac * M
    Phi = M - mstar
    kf = float(_kf(n0 / 2.0))
    ef = math.sqrt(kf**2 + mstar**2)
    W0 = (M_mev + BA) / HBARC - ef                      # p = 0: E/A = E*_F + W
    Co = W0 / n0

    def rho_s(n, ms):
        return 2.0 * _rho_s_one(_kf(n / 2.0), ms)

    rs0 = float(rho_s(n0, mstar))
    drs_dn = float(jax.grad(rho_s, 0)(n0, mstar))
    drs_dm = float(jax.grad(rho_s, 1)(n0, mstar))
    eps_kin = 2.0 * float(_eps_kin_one(kf, mstar))
    U0 = n0 * (M_mev + BA) / HBARC - eps_kin - 0.5 * Co * n0**2
    dkf = kf / (3.0 * n0)
    # K = 9 n0 dmu/dn, mu = E*_F(n) + C_omega n  ->  dPhi/dn
    dPhi = (kf * dkf - ef * (K / HBARC / (9.0 * n0) - Co)) / mstar
    Upp = drs_dn / dPhi - drs_dm                        # U''(Phi) from d(gap)/dn
    A = np.array([[Phi, M * Phi**2, Phi**3],
                  [Phi**2 / 2.0, M * Phi**3 / 3.0, Phi**4 / 4.0],
                  [1.0, 2.0 * M * Phi, 3.0 * Phi**2]])
    inv_cs, b, c = np.linalg.solve(A, np.array([rs0, U0, Upp]))
    if inv_cs <= 0.0:
        raise ValueError("saturation point needs C_sigma <= 0")
    T = kf**2 / (6.0 * ef)                              # kinetic part of E_sym
    cr_eff = 8.0 * (J / HBARC - T) / n0
    if cr_eff <= 0.0:
        raise ValueError("J below its kinetic part; C_rho <= 0")
    def_dn = (kf * dkf - mstar * dPhi) / ef             # dE*_F/dn (m* = M - Phi)
    dT = kf * dkf / (3.0 * ef) - T * def_dn / ef
    J_rho = cr_eff * n0 / 8.0
    L0 = 3.0 * n0 * dT * HBARC + 3.0 * J_rho * HBARC
    lam, Cr = 0.0, cr_eff
    if L is not None:
        q = (L0 - L) / (3.0 * J_rho * HBARC)
        if not (0.0 <= q < 2.0):
            raise ValueError(f"L={L} outside ({L0 - 6 * J_rho * HBARC:.1f}, "
                             f"{L0:.1f}] for this saturation point")
        x = q / (2.0 - q)                               # x = 2 Lambda_v C_rho W0^2
        Cr = cr_eff * (1.0 + x)
        lam = x / (2.0 * Cr * W0**2)
    return dict(M_mev=M_mev, C_sigma=1.0 / inv_cs, C_omega=Co, C_rho=Cr,
                b=b, c=c, Lambda_v=lam, L0=L0)


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
                 n_surface=1e-7, gamma_crust=1.3,
                 crust="sly", n_t=N_T_DEFAULT,
                 allow_unstable_core=False):
        self.n_grid = jnp.exp(jnp.linspace(math.log(n_min), math.log(n_max),
                                           n_points))
        self.n_surface = n_surface
        self.gamma_crust = gamma_crust
        self.crust = check_crust(crust, n_surface)
        self.n_t = n_t
        self.allow_unstable_core = allow_unstable_core

    def _table_grid(self):
        """Uniform-matter nodes of the EOS table (above n_t with the SLy crust)."""
        if self.crust == "sly" and self.n_surface is not None:
            return core_grid(self.n_grid, self.n_t)
        return self.n_grid

    # --- symmetric nuclear matter (for saturation-property validation) ---
    def symmetric(self, n_B, params=None):
        p = dict(DEFAULT_PARAMS)
        if params:
            p.update(params)
        eps, pres, _ = _eps_p_of_n(jnp.asarray(n_B), 0.5, p, leptons=False)
        return eps, pres        # MeV/fm^3

    def composition(self, params: dict = None) -> Composition:
        """beta-equilibrium (x_p, x_e, x_mu) on ``self.n_grid``, the same solve
        that builds :meth:`eos_table`."""
        p = dict(DEFAULT_PARAMS)
        if params:
            p.update(params)
        n = self.n_grid
        x_p = _solve_xp(n, p)
        _, n_e, n_mu, _ = _leptons(x_p * n)
        return Composition(n=n, x_p=x_p, x_e=n_e / n, x_mu=n_mu / n)

    def landau_masses(self, params, comp):
        """(m*_n/M, m*_p/M) on ``comp`` (see :func:`landau_masses`)."""
        return landau_masses(params, comp.n, comp.x_p)

    def eos_table(self, params: dict = None):
        p = dict(DEFAULT_PARAMS)
        if params:
            p.update(params)
        n = self._table_grid()
        x_p = _solve_xp(n, p)                              # beta-equilibrium
        Phi = _solve_phi((1.0 - x_p) * n, x_p * n, p)      # robust first-root, vec
        eps, pres, _ = jax.vmap(lambda ni, xi, Ph: _eps_p_from_state(ni, xi, Ph, p))(
            n, x_p, Phi)
        pres, ln_nu = monotone_core(n, pres, self.n_t)
        return finish_table(n, eps, pres, self.crust, self.n_t,
                            self.n_surface, self.gamma_crust, ln_nu,
                            self.allow_unstable_core)
