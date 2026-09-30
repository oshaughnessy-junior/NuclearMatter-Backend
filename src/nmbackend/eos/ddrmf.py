"""Density-dependent relativistic mean field (DD-RMF), Typel-Wolter form.

Nucleons couple to sigma, omega and rho mesons with couplings that depend on
the baryon density n (Typel & Wolter, NPA 656, 331 (1999)):

    Gamma_i(n) = Gamma_i(n_ref) f_i(x),   x = n / n_ref
    f_i(x) = a_i (1 + b_i (x+d_i)^2) / (1 + c_i (x+d_i)^2)   (i = sigma, omega)
    f_rho(x) = exp(-a_rho (x - 1))

There are no meson self-interactions.  With C_i(n) = (Gamma_i(n)/m_i)^2 (fm^2),
Phi = Gamma_sigma sigma, n_3 = n_p - n_n and isospin t_p = +1, t_n = -1
(the convention of DD2 and DDME2)::

    m*_i  = M_i - Phi,        Phi = C_sigma(n) n_s          (self-consistent)
    eps   = sum_i eps_kin(kF_i, m*_i) + Phi^2 / (2 C_sigma)
            + C_omega n^2 / 2 + C_rho n_3^2 / 2 + eps_leptons
    mu_i  = E*_Fi + C_omega n + t_i C_rho n_3 + Sigma_R
    Sigma_R = -C_sigma' n_s^2 / 2 + C_omega' n^2 / 2 + C_rho' n_3^2 / 2

Sigma_R is the rearrangement term (C' = dC/dn).  Without it p = sum mu_i n_i
- eps is not the thermodynamic pressure n^2 d(eps/n)/dn.

beta equilibrium and charge neutrality use npe-mu leptons
(:func:`nmbackend.eos.leptons.lepton_sector`).

``couplings_from_saturation_dd`` maps (n0, B/A, K, m*/M, J, L) to couplings,
with c_i = 1/(3 d_i^2) (f''(0) = 0).  The shape parameters d_sigma, d_omega
default to their DD2 values and may be passed freely.  DD2 is a member of
that family.  The rho decay a_rho follows from L at fixed J; a_rho < 0 (rho
coupling growing with density, L above its a_rho = 0 value) is rejected
unless ``allow_negative_arho=True``.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from ..constants import HBARC
from .base import EOSModel, Composition, compiled_table
from .leptons import lepton_sector
from .crust import (N_T_DEFAULT, check_crust, core_grid, finish_table_flags,
                    monotone_core)

jax.config.update("jax_enable_x64", True)

_PI2 = math.pi ** 2


def _set(M_n, M_p, n_ref, m_s, m_w, m_r, G_s, G_w, G_r, s, w, a_r):
    """Parameter dict from published (masses, couplings, TW coefficients)."""
    C = lambda G, m: (G * HBARC / m) ** 2                     # fm^2
    return dict(M_n=M_n, M_p=M_p, n_ref=n_ref,
                C_sigma=C(G_s, m_s), C_omega=C(G_w, m_w), C_rho=C(G_r, m_r),
                a_s=s[0], b_s=s[1], c_s=s[2], d_s=s[3],
                a_w=w[0], b_w=w[1], c_w=w[2], d_w=w[3], a_r=a_r)


#: DD2: Typel, Roepke, Klaehn, Blaschke & Wolter, PRC 81, 015803 (2010),
#: arXiv:0908.2344, Tables II-III.
DD2 = _set(939.56536, 938.27203, 0.149065, 546.212459, 783.0, 763.0,
           10.686681, 13.342362, 3.626940,
           (1.357630, 0.634442, 1.005358, 0.575810),
           (1.369718, 0.496475, 0.817753, 0.638452), 0.518903)

#: DDME2: Lalazissis, Niksic, Vretenar & Ring, PRC 71, 024312 (2005), Table I.
#: Single nucleon mass 939 MeV.
DDME2 = _set(939.0, 939.0, 0.152, 550.1238, 783.0, 763.0,
             10.5396, 13.0189, 3.6836,
             (1.3881, 1.0943, 1.7057, 0.4421),
             (1.3892, 0.9240, 1.4620, 0.4775), 0.5647)

DEFAULT_PARAMS = dict(DD2)


# --- coupling functions C_i(n) and dC_i/dn ---------------------------------
def _tw(x, a, b, c, d):
    y = (x + d) ** 2
    f = a * (1.0 + b * y) / (1.0 + c * y)
    fp = 2.0 * a * (b - c) * (x + d) / (1.0 + c * y) ** 2
    return f, fp


def _couplings(n, p):
    """(C_sigma, C_omega, C_rho) and their n-derivatives at density n."""
    x = n / p["n_ref"]
    fs, fsp = _tw(x, p["a_s"], p["b_s"], p["c_s"], p["d_s"])
    fw, fwp = _tw(x, p["a_w"], p["b_w"], p["c_w"], p["d_w"])
    fr = jnp.exp(-p["a_r"] * (x - 1.0))
    fr_p = -p["a_r"] * fr
    C = (p["C_sigma"] * fs**2, p["C_omega"] * fw**2, p["C_rho"] * fr**2)
    dC = (2.0 * p["C_sigma"] * fs * fsp / p["n_ref"],
          2.0 * p["C_omega"] * fw * fwp / p["n_ref"],
          2.0 * p["C_rho"] * fr * fr_p / p["n_ref"])
    return C, dC


# --- Fermi-gas integrals (degeneracy 2, fm units) --------------------------
def _kf(n_i):
    return (3.0 * _PI2 * jnp.maximum(n_i, 1e-30)) ** (1.0 / 3.0)


def _rho_s(kf, ms):
    ef = jnp.sqrt(kf**2 + ms**2)
    return (ms / (2.0 * _PI2)) * (kf * ef - ms**2 * jnp.log((kf + ef) / ms))


def _eps_kin(kf, ms):
    ef = jnp.sqrt(kf**2 + ms**2)
    return (1.0 / (8.0 * _PI2)) * (
        kf * (2.0 * kf**2 + ms**2) * ef - ms**4 * jnp.log((kf + ef) / ms))


def _solve_phi(n_n, n_p, p, nbis=60):
    """Phi = C_sigma(n) n_s(M - Phi): the residual is monotone in Phi, so
    bisection on [0, M_min) finds the unique root; a final Newton step from a
    stop-gradient start gives the implicit derivative."""
    Mn, Mp = p["M_n"] / HBARC, p["M_p"] / HBARC
    kfn, kfp = _kf(n_n), _kf(n_p)
    Cs = _couplings(n_n + n_p, p)[0][0]

    def resid(Phi):
        return Phi - Cs * (_rho_s(kfn, Mn - Phi) + _rho_s(kfp, Mp - Phi))

    lo = jnp.zeros_like(kfn)
    hi = 0.999 * jnp.minimum(Mn, Mp) + 0.0 * kfn

    def body(_i, c):
        lo, hi = c
        mid = 0.5 * (lo + hi)
        r = resid(mid)
        return jnp.where(r < 0.0, mid, lo), jnp.where(r < 0.0, hi, mid)

    lo, hi = jax.lax.fori_loop(0, nbis, body, (lo, hi))
    x0 = jax.lax.stop_gradient(0.5 * (lo + hi))
    d = jax.grad(lambda x: jnp.sum(resid(x)))(x0)
    return x0 - resid(x0) / d


def _nucleon_state(n_n, n_p, p):
    """Nucleon eps (fm^-4) and mu_n, mu_p (fm^-1)."""
    Mn, Mp = p["M_n"] / HBARC, p["M_p"] / HBARC
    n = n_n + n_p
    n3 = n_p - n_n
    Phi = _solve_phi(n_n, n_p, p)
    kfn, kfp = _kf(n_n), _kf(n_p)
    msn, msp = Mn - Phi, Mp - Phi
    ns = _rho_s(kfn, msn) + _rho_s(kfp, msp)
    (Cs, Cw, Cr), (dCs, dCw, dCr) = _couplings(n, p)
    eps = (_eps_kin(kfn, msn) + _eps_kin(kfp, msp) + 0.5 * Phi**2 / Cs
           + 0.5 * Cw * n**2 + 0.5 * Cr * n3**2)
    sig_r = -0.5 * dCs * ns**2 + 0.5 * dCw * n**2 + 0.5 * dCr * n3**2
    base = Cw * n + sig_r
    mu_n = jnp.sqrt(kfn**2 + msn**2) + base - Cr * n3
    mu_p = jnp.sqrt(kfp**2 + msp**2) + base + Cr * n3
    return eps, mu_n, mu_p, Phi


def landau_masses(params, n, x_p):
    """(m*_n/M_n, m*_p/M_p): Landau masses at the Fermi surface.

    The single-particle energy is sqrt(k^2 + m_D^2) + (vector and
    rearrangement self-energies, independent of k), so the Landau mass
    k_F / (d e/dk) is E*_F = sqrt(k_F^2 + m_D^2), with Dirac mass
    m_D = M_i - Phi."""
    p = dict(DEFAULT_PARAMS)
    p.update({k: v for k, v in (params or {}).items() if k != "L0"})
    n = jnp.asarray(n, float)
    x_p = jnp.asarray(x_p, float)
    n_n, n_p = (1.0 - x_p) * n, x_p * n
    Phi = _solve_phi(n_n, n_p, p)
    Mn, Mp = p["M_n"] / HBARC, p["M_p"] / HBARC
    ef = lambda kf, m: jnp.sqrt(kf**2 + (m - Phi) ** 2) / m
    return ef(_kf(n_n), Mn), ef(_kf(n_p), Mp)


def _solve_xp(n_B, p, n_iter=60):
    """beta-equilibrium proton fraction (npe-mu): mu_n - mu_p = mu_e."""
    def g(x_p):
        _, mu_n, mu_p, _ = _nucleon_state((1.0 - x_p) * n_B, x_p * n_B, p)
        return (mu_n - mu_p) - lepton_sector(x_p * n_B, muons=True)[0]

    lo = jnp.full_like(n_B, 1e-5)
    hi = jnp.full_like(n_B, 0.5)

    def body(_, c):
        lo, hi = c
        mid = 0.5 * (lo + hi)
        gm = g(mid)                        # decreasing in x_p
        return jnp.where(gm > 0.0, mid, lo), jnp.where(gm > 0.0, hi, mid)

    lo, hi = jax.lax.fori_loop(0, n_iter, body, (lo, hi))
    x0 = jax.lax.stop_gradient(0.5 * (lo + hi))
    dg = jax.grad(lambda x: jnp.sum(g(x)))(x0)
    return x0 - g(x0) / dg


def _eps_p(n_B, x_p, p, leptons=True):
    """eps, p (MeV/fm^3) at (n_B, x_p); p = sum_i mu_i n_i - eps."""
    n_p, n_n = x_p * n_B, (1.0 - x_p) * n_B
    eps, mu_n, mu_p, _ = _nucleon_state(n_n, n_p, p)
    pres = mu_n * n_n + mu_p * n_p - eps
    if leptons:
        mu_e, n_e, n_mu, eps_l = lepton_sector(n_p, muons=True)
        eps = eps + eps_l
        pres = pres + mu_e * (n_e + n_mu) - eps_l
    return eps * HBARC, pres * HBARC


def symmetric_matter(n, params=None):
    """Symmetric nuclear matter (no leptons): dict of E/A - M_avg (MeV),
    p (MeV/fm^3), m*/M_avg, S(n) (MeV) on density array ``n`` (fm^-3).
    S uses the curvature form S = kF^2 / (6 E*_F) + C_rho n / 2."""
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update({k: v for k, v in params.items() if k != "L0"})
    return _symmetric(jnp.asarray(n, float), p)


@jax.jit
def _symmetric(n, p):
    eps, mu_n, mu_p, Phi = _nucleon_state(n / 2.0, n / 2.0, p)
    M = 0.5 * (p["M_n"] + p["M_p"]) / HBARC
    pres = 0.5 * (mu_n + mu_p) * n - eps
    kf = _kf(n / 2.0)
    ms = M - Phi
    Cr = _couplings(n, p)[0][2]
    S = kf**2 / (6.0 * jnp.sqrt(kf**2 + ms**2)) + 0.5 * Cr * n
    return dict(EA=(eps / n - M) * HBARC, p=pres * HBARC, mstar=ms / M,
                S=S * HBARC)


def saturation_properties(params=None, n_lo=0.10, n_hi=0.20, npts=2001):
    """n0, B/A, K, m*/M, E_sym (= J), L of a DD-RMF parameter set."""
    ns = np.linspace(n_lo, n_hi, npts)
    s = {k: np.asarray(v) for k, v in symmetric_matter(ns, params).items()}
    i = int(np.argmin(s["EA"]))
    dn = ns[1] - ns[0]
    # refine the minimum with a parabola through three nodes
    e0, e1, e2 = s["EA"][i - 1:i + 2]
    t = 0.5 * (e0 - e2) / (e0 - 2 * e1 + e2)
    n0 = ns[i] + t * dn
    q = lambda a: float(np.interp(n0, ns, a))
    d = lambda a: np.gradient(a, dn)
    K = 9.0 * q(d(s["p"]))
    L = 3.0 * n0 * q(d(s["S"]))
    return dict(n0=float(n0), BA=q(s["EA"]), K=K, mstar_frac=q(s["mstar"]),
                E_sym=q(s["S"]), L=L)


def _tw_from_b(b, d):
    """(a, b, c, d) with f(1) = 1 and f''(0) = 0 for given b, d."""
    c = 1.0 / (3.0 * d * d)
    a = (1.0 + c * (1.0 + d) ** 2) / (1.0 + b * (1.0 + d) ** 2)
    return a, b, c, d


def couplings_from_saturation_dd(n0=0.149065, BA=-16.02, K=242.7,
                                 mstar_frac=0.5628, J=31.7, L=55.0,
                                 M_mev=938.9187, d_s=DD2["d_s"],
                                 d_w=DD2["d_w"], b_start=(DD2["b_s"], DD2["b_w"]),
                                 allow_negative_arho=False):
    """DD-RMF couplings reproducing a saturation point.

    n0 in fm^-3; BA, K, J, L in MeV.  C_sigma(n0) from m*/M, C_omega(n0) from
    B/A, (b_sigma, b_omega) from p(n0) = 0 and K (2-d root), C_rho(n0) from J
    and a_rho from L.  Single nucleon mass ``M_mev``.

    Raises ValueError if no solution exists, if a coupling function turns
    negative or singular on 0 <= x <= 12, or (unless ``allow_negative_arho``)
    if a_rho < 0, a rho coupling that grows with density.
    """
    from scipy.optimize import fsolve

    M = M_mev / HBARC
    ms = mstar_frac * M
    Phi0 = M - ms
    kf = float(_kf(n0 / 2.0))
    ef = math.sqrt(kf**2 + ms**2)
    ns0 = float(2.0 * _rho_s(kf, ms))
    Cs0 = Phi0 / ns0
    ekin = float(2.0 * _eps_kin(kf, ms))
    Cw0 = 2.0 * (n0 * (M_mev + BA) / HBARC - ekin - 0.5 * Phi0**2 / Cs0) / n0**2
    if Cs0 <= 0.0 or Cw0 <= 0.0:
        raise ValueError("saturation point needs a non-positive coupling")

    def params(b):
        s, w = _tw_from_b(b[0], d_s), _tw_from_b(b[1], d_w)
        return dict(M_n=M_mev, M_p=M_mev, n_ref=n0, C_sigma=Cs0, C_omega=Cw0,
                    C_rho=1.0, a_s=s[0], b_s=s[1], c_s=s[2], d_s=s[3],
                    a_w=w[0], b_w=w[1], c_w=w[2], d_w=w[3], a_r=0.0)

    h = 2e-4 * n0
    grid = jnp.array([n0 - h, n0, n0 + h])

    def resid(b):
        pr = np.asarray(symmetric_matter(grid, params(b))["p"])
        return [pr[1] / 1.0, (9.0 * (pr[2] - pr[0]) / (2 * h) - K) / K]

    b, info, ier, _ = fsolve(resid, b_start, full_output=True, xtol=1e-11)
    if ier != 1 or max(abs(v) for v in resid(b)) > 1e-6:
        raise ValueError("no (b_sigma, b_omega) reproduces p(n0) = 0 and K")
    p = params(b)
    x = np.linspace(0.0, 12.0, 241)
    for k in ("s", "w"):
        f, _ = _tw(x, p["a_" + k], p["b_" + k], p["c_" + k], p["d_" + k])
        den = 1.0 + p["c_" + k] * (x + p["d_" + k]) ** 2
        if not (np.all(np.asarray(f) > 0.0) and np.all(den > 0.0)):
            raise ValueError("coupling function not positive on 0 <= x <= 12")
    # isovector: J -> C_rho(n0); L -> a_rho
    T = kf**2 / (6.0 * ef)
    Srho = J / HBARC - T
    if Srho <= 0.0:
        raise ValueError("J below its kinetic part")
    p["C_rho"] = 2.0 * Srho / n0
    S = np.asarray(symmetric_matter(grid, p)["S"])       # a_r = 0 here
    L_a0 = 3.0 * n0 * (S[2] - S[0]) / (2 * h)            # L at a_rho = 0
    a_r = (L_a0 - L) / (6.0 * Srho * HBARC)               # dL/da_rho = -6 S_rho
    if a_r < 0.0 and not allow_negative_arho:
        raise ValueError(f"L={L} above the a_rho = 0 value {L_a0:.1f}")
    p["a_r"] = float(a_r)
    p.update(L0=float(L_a0))
    return p


class DDRMFEOS(EOSModel):
    """Density-dependent RMF (Typel-Wolter), npe-mu beta equilibrium."""

    name = "DDRMFEOS"

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

    @staticmethod
    def _p(params):
        p = dict(DEFAULT_PARAMS)
        if params:
            p.update({k: v for k, v in params.items() if k != "L0"})
        return p

    def composition(self, params: dict = None) -> Composition:
        """beta-equilibrium (x_p, x_e, x_mu) on ``self.n_grid``."""
        p = self._p(params)
        n = self.n_grid
        x_p = _solve_xp(n, p)
        _, n_e, n_mu, _ = lepton_sector(x_p * n, muons=True)
        return Composition(n=n, x_p=x_p, x_e=n_e / n, x_mu=n_mu / n)

    def landau_masses(self, params, comp):
        """(m*_n/M_n, m*_p/M_p) on ``comp`` (see :func:`landau_masses`)."""
        return landau_masses(self._p(params), comp.n, comp.x_p)

    def eos_table(self, params: dict = None):
        return compiled_table(self, self._p(params))

    def _table_flags(self, p):
        n = self._table_grid()
        x_p = _solve_xp(n, p)
        eps, pres = _eps_p(n, x_p, p)
        pres, ln_nu = monotone_core(n, pres, self.n_t)
        return finish_table_flags(n, eps, pres, self.crust, self.n_t,
                                  self.n_surface, self.gamma_crust, ln_nu,
                                  self.allow_unstable_core)
