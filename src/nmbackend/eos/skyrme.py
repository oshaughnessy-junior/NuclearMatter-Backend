"""Standard Skyrme energy density functional for uniform npe-mu matter.

Homogeneous nuclear matter depends on the Skyrme parameters only through
seven combinations.  With tau_q = (3/5)(3 pi^2)^(2/3) n_q^(5/3) and
tau = tau_n + tau_p (Chabanat et al., NPA 627, 710 (1997))::

    H = h2m tau + a0 n^2 + b0 (n_n^2 + n_p^2)
        + [a3 n^2 + b3 (n_n^2 + n_p^2)] n^sigma
        + c1 n tau + c2 (n_n tau_n + n_p tau_p)

    a0 = t0 (1 + x0/2) / 2        b0 = -t0 (x0 + 1/2) / 2
    a3 = t3 (1 + x3/2) / 12       b3 = -t3 (x3 + 1/2) / 12
    c1 = [t1 (2 + x1) + t2 (2 + x2)] / 8
    c2 = [t2 (2 x2 + 1) - t1 (2 x1 + 1)] / 8

H is in MeV/fm^3 without rest mass; h2m = hbar^2 / 2m.  Chemical potentials
are mu_q = M_q + dH/dn_q (autodiff), p = sum_q mu_q n_q - eps.
beta equilibrium uses npe-mu leptons (:func:`nmbackend.eos.leptons.lepton_sector`).

``skyrme_from_saturation`` inverts (n0, E0, K, m*_s, m*_v, J, L) to the seven
combinations in closed form.  The effective masses are
h2m / m*_s = h2m + (c1 + c2/2) n (symmetric matter) and
h2m / m*_v = h2m + c1 n (a proton in neutron matter), in units of m.

Extended Skyrme (BSk form; Chamel, Goriely & Pearson, PRC 80, 065804 (2009);
BSkG3, Grams et al., arXiv:2307.14276, Eq. 8) adds t4 and t5 terms, which in
uniform matter act as density-dependent t1 and t2:
c1 -> c1 + c1b n^beta + c1g n^gamma, and likewise c2.  The effective masses
then depend on density beyond the 1/n form, which frees S(n) above
saturation (K_sym is no longer fixed by J, L, m*).  All four extended
coefficients default to zero (standard Skyrme).

The Skyrme form is non-relativistic.  At high density it can become acausal
(cs2 > 1) and the neutron effective mass can turn negative; the table does
not correct either, and the caller decides what to do (see ``diagnostics``).
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from ..constants import HBARC, M_N, M_P
from .base import EOSModel, Composition
from .leptons import lepton_sector
from .crust import (N_T_DEFAULT, check_crust, core_grid, finish_table,
                    monotone_core)

jax.config.update("jax_enable_x64", True)

_PI2 = math.pi ** 2
_CTAU = 0.6 * (3.0 * _PI2) ** (2.0 / 3.0)


def skyrme_from_t(t0, t1, t2, t3, x0, x1, x2, x3, sigma, h2m=20.73553):
    """Uniform-matter combinations from standard Skyrme parameters."""
    return dict(h2m=h2m, sigma=sigma,
                a0=0.5 * t0 * (1.0 + 0.5 * x0), b0=-0.5 * t0 * (x0 + 0.5),
                a3=t3 * (1.0 + 0.5 * x3) / 12.0, b3=-t3 * (x3 + 0.5) / 12.0,
                c1=(t1 * (2.0 + x1) + t2 * (2.0 + x2)) / 8.0,
                c2=(t2 * (2.0 * x2 + 1.0) - t1 * (2.0 * x1 + 1.0)) / 8.0)


#: SLy4: Chabanat, Bonche, Haensel, Meyer & Schaeffer, NPA 635, 231 (1998).
SLY4 = skyrme_from_t(t0=-2488.91, t1=486.82, t2=-546.39, t3=13777.0,
                     x0=0.834, x1=-0.344, x2=-1.0, x3=1.354, sigma=1.0 / 6.0)

#: Extended-Skyrme coefficients (zero: standard Skyrme).
EXT_ZERO = dict(c1b=0.0, c2b=0.0, c1g=0.0, c2g=0.0, beta=1.0 / 12.0,
                gamma=0.25)


def skyrme_from_t_ext(t0, t1, t2, t3, t4, t5, x0, x1, x2t2, x3, x4, x5,
                      sigma, beta, gamma, h2m=20.73553):
    """Uniform-matter combinations of the extended (BSk) Skyrme form.

    t4 enters like t1 times n^beta, t5 like t2 times n^gamma.  As in the BSk
    tables, x2 is given through the product x2 t2."""
    p = skyrme_from_t(t0, t1, 0.0, t3, x0, x1, 0.0, x3, sigma, h2m)
    p["c1"] += (2.0 * t2 + x2t2) / 8.0
    p["c2"] += (t2 + 2.0 * x2t2) / 8.0
    p.update(c1b=t4 * (2.0 + x4) / 8.0, c2b=-t4 * (2.0 * x4 + 1.0) / 8.0,
             c1g=t5 * (2.0 + x5) / 8.0, c2g=t5 * (2.0 * x5 + 1.0) / 8.0,
             beta=beta, gamma=gamma)
    return p


#: BSkG3: Grams, Ryssens, Scamps, Goriely & Chamel, arXiv:2307.14276,
#: Table 1 (single h2m; the paper uses separate n and p masses).
BSKG3 = skyrme_from_t_ext(t0=-2325.35, t1=749.82, t2=0.01, t3=14083.45,
                          t4=-498.01, t5=266.52, x0=0.558834, x1=2.940880,
                          x2t2=-432.256954, x3=0.628699, x4=5.657990,
                          x5=0.396933, sigma=0.2, beta=1.0 / 12.0, gamma=0.25)

DEFAULT_PARAMS = dict(SLY4, **EXT_ZERO)


def _c12(n, p):
    """Density-dependent (c1, c2) of the extended form."""
    if "c1b" not in p:                       # standard-Skyrme dict
        return p["c1"], p["c2"]
    nb, ng = n ** p["beta"], n ** p["gamma"]
    return (p["c1"] + p["c1b"] * nb + p["c1g"] * ng,
            p["c2"] + p["c2b"] * nb + p["c2g"] * ng)


def _H(n_n, n_p, p):
    """Nuclear energy density without rest mass (MeV/fm^3)."""
    n = n_n + n_p
    tn = _CTAU * jnp.maximum(n_n, 0.0) ** (5.0 / 3.0)
    tp = _CTAU * jnp.maximum(n_p, 0.0) ** (5.0 / 3.0)
    s2 = n_n**2 + n_p**2
    c1, c2 = _c12(n, p)
    return (p["h2m"] * (tn + tp) + p["a0"] * n**2 + p["b0"] * s2
            + (p["a3"] * n**2 + p["b3"] * s2) * n ** p["sigma"]
            + c1 * n * (tn + tp) + c2 * (n_n * tn + n_p * tp))


def _nucleon(n_n, n_p, p):
    """eps (MeV/fm^3, with rest mass), mu_n, mu_p (MeV)."""
    H = _H(n_n, n_p, p)
    dHn = jax.grad(lambda a, b: jnp.sum(_H(a, b, p)), 0)(n_n, n_p)
    dHp = jax.grad(lambda a, b: jnp.sum(_H(a, b, p)), 1)(n_n, n_p)
    return H + n_n * M_N + n_p * M_P, dHn + M_N, dHp + M_P


def _solve_xp(n_B, p, n_iter=60):
    """beta-equilibrium proton fraction (npe-mu): mu_n - mu_p = mu_e."""
    def g(x_p):
        _, mu_n, mu_p = _nucleon((1.0 - x_p) * n_B, x_p * n_B, p)
        return (mu_n - mu_p) - HBARC * lepton_sector(x_p * n_B, muons=True)[0]

    lo = jnp.full_like(n_B, 1e-6)
    hi = jnp.full_like(n_B, 0.5)

    def body(_, c):
        lo, hi = c
        mid = 0.5 * (lo + hi)
        gm = g(mid)
        return jnp.where(gm > 0.0, mid, lo), jnp.where(gm > 0.0, hi, mid)

    lo, hi = jax.lax.fori_loop(0, n_iter, body, (lo, hi))
    x0 = jax.lax.stop_gradient(0.5 * (lo + hi))
    dg = jax.grad(lambda x: jnp.sum(g(x)))(x0)
    return x0 - g(x0) / dg


def _eps_p(n_B, x_p, p):
    n_p, n_n = x_p * n_B, (1.0 - x_p) * n_B
    eps, mu_n, mu_p = _nucleon(n_n, n_p, p)
    mu_e, n_e, n_mu, eps_l = lepton_sector(n_p, muons=True)
    pres = mu_n * n_n + mu_p * n_p - eps + HBARC * (mu_e * (n_e + n_mu) - eps_l)
    return eps + HBARC * eps_l, pres


def symmetric_matter(n, params=None):
    """E/A (MeV, no rest mass), p (MeV/fm^3), S(n) (MeV, curvature) for
    symmetric matter on ``n`` (fm^-3)."""
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    n = jnp.asarray(n, float)
    ea = lambda nn, d: _H(nn * (1 + d) / 2, nn * (1 - d) / 2, p) / nn
    EA = ea(n, 0.0)
    dEA = jax.vmap(jax.grad(ea, 0), (0, None))(n, 0.0)
    S = 0.5 * jax.vmap(jax.grad(jax.grad(ea, 1), 1), (0, None))(n, 0.0)
    return dict(EA=EA, p=n**2 * dEA, S=S)


def saturation_properties(params=None, n_lo=0.10, n_hi=0.22, npts=2401):
    """n0, B/A, K, E_sym (= J), L, K_sym, m*_s/m, m*_v/m of a Skyrme set."""
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    ns = np.linspace(n_lo, n_hi, npts)
    s = {k: np.asarray(v) for k, v in symmetric_matter(ns, p).items()}
    i = int(np.argmin(s["EA"]))
    dn = ns[1] - ns[0]
    e0, e1, e2 = s["EA"][i - 1:i + 2]
    n0 = ns[i] + 0.5 * dn * (e0 - e2) / (e0 - 2 * e1 + e2)
    q = lambda a: float(np.interp(n0, ns, a))
    d = lambda a: np.gradient(a, dn)
    ms, mv = effective_masses(p, n0)
    return dict(n0=float(n0), BA=q(s["EA"]), K=9.0 * q(d(s["p"])),
                E_sym=q(s["S"]), L=3.0 * n0 * q(d(s["S"])),
                K_sym=9.0 * n0**2 * q(d(d(s["S"]))),
                ms_frac=float(ms), mv_frac=float(mv))


def effective_masses(params, n):
    """(m*_s/m, m*_v/m) at density n (symmetric matter)."""
    p = dict(DEFAULT_PARAMS)
    p.update(params)
    c1, c2 = _c12(np.asarray(n, float), p)
    return (p["h2m"] / (p["h2m"] + (c1 + 0.5 * c2) * n),
            p["h2m"] / (p["h2m"] + c1 * n))


def skyrme_from_saturation(n0=0.160, BA=-15.97, K=230.0, ms_frac=0.70,
                           mv_frac=0.80, J=32.0, L=46.0, h2m=20.73553,
                           ms_hi=None, mv_hi=None, n_hi=(2.0, 4.0),
                           beta=1.0 / 12.0, gamma=0.25):
    """Uniform-matter Skyrme combinations reproducing a saturation point.

    With T(n) = (3/5) h2m kF^2 (symmetric) and u(n) = c1 + c2/2,
    E/A = T (1 + u n / h2m) + A n + B n^(sigma+1); E/A(n0) = BA and
    p(n0) = 0 fix A, B for given sigma, and K fixes sigma.  The isovector part
    is linear in (b0, b3) given c1(n), c2(n).  Raises ValueError if
    sigma <= 0.

    Standard Skyrme: c1, c2 constant, fixed by m*_s, m*_v at n0.  Extended
    Skyrme: pass ``ms_hi`` and ``mv_hi``, the effective masses (units of m) at
    the two densities ``n_hi`` (units of n0); with the n0 values they fix the
    constant, n^beta and n^gamma parts of u(n) and c1(n) (3x3 linear solves).
    """
    kf2 = lambda n: (1.5 * _PI2 * n) ** (2.0 / 3.0)
    T = lambda n: 0.6 * h2m * kf2(n)
    ext = dict(EXT_ZERO, beta=beta, gamma=gamma)
    if ms_hi is None:
        u0 = (h2m / ms_frac - h2m) / n0
        v0 = (h2m / mv_frac - h2m) / n0
    else:
        ns = n0 * np.array([1.0, n_hi[0], n_hi[1]])
        Mx = np.stack([np.ones(3), ns**beta, ns**gamma], axis=1)
        ru = (h2m / np.array([ms_frac, ms_hi[0], ms_hi[1]]) - h2m) / ns
        rv = (h2m / np.array([mv_frac, mv_hi[0], mv_hi[1]]) - h2m) / ns
        u0, ub, ug = np.linalg.solve(Mx, ru)
        v0, vb, vg = np.linalg.solve(Mx, rv)
        ext.update(c1b=vb, c2b=2.0 * (ub - vb), c1g=vg, c2g=2.0 * (ug - vg))
    c1, c2 = v0, 2.0 * (u0 - v0)
    pc = dict(ext, c1=c1, c2=c2)

    def kin(n):          # kinetic E/A in symmetric matter
        a, b = _c12(n, pc)
        return T(n) * (1.0 + (a + 0.5 * b) * n / h2m)

    def skin(n):         # kinetic S(n)
        a, b = _c12(n, pc)
        return (5.0 / 9.0) * T(n) * (1.0 + (a + 2.0 * b) * n / h2m)

    h = 1e-3 * n0
    d1 = lambda f: (f(n0 + h) - f(n0 - h)) / (2.0 * h)
    d2 = lambda f: (f(n0 + h) - 2.0 * f(n0) + f(n0 - h)) / h**2
    Tk = kin(n0)
    Q = n0 * d1(kin) + BA - Tk
    KT = 9.0 * n0**2 * d2(kin)
    sigma = (KT - K) / (9.0 * Q) - 1.0
    if not sigma > 0.0:
        raise ValueError(f"no Skyrme sigma > 0 for K={K} (sigma={sigma:.3f})")
    B = -Q / (sigma * n0 ** (sigma + 1.0))
    A = (BA - Tk - B * n0 ** (sigma + 1.0)) / n0
    # isovector: S = S_kin(n) + b0 n/2 + b3 n^(s+1)/2
    Sk, dSk = skin(n0), d1(skin)
    y = n0 ** (sigma + 1.0) / 2.0
    b3 = (L / 3.0 - n0 * dSk - J + Sk) / (sigma * y)
    b0 = 2.0 * (J - Sk - b3 * y) / n0
    out = dict(h2m=h2m, sigma=float(sigma), a0=A - 0.5 * b0, b0=b0,
               a3=B - 0.5 * b3, b3=b3, c1=c1, c2=c2)
    if ms_hi is not None:
        out.update({k: float(v) for k, v in ext.items()})
    return out


def landau_masses(params, n, x_p):
    """(m*_n/m, m*_p/m) in matter of density n and proton fraction x_p.

    The Skyrme single-particle energy is quadratic in k, so the Landau mass
    at the Fermi surface is h2m / m*_q = h2m + c1(n) n + c2(n) n_q.  In units
    of the h2m mass (one mass for n and p)."""
    p = dict(DEFAULT_PARAMS)
    p.update(params or {})
    n = jnp.asarray(n, float)
    x_p = jnp.asarray(x_p, float)
    c1, c2 = _c12(n, p)
    em = lambda nq: p["h2m"] / (p["h2m"] + c1 * n + c2 * nq)
    return em((1.0 - x_p) * n), em(x_p * n)


def diagnostics(params, n):
    """Per-density checks on beta-equilibrium matter: minimum of
    h2m/m*_q (MeV fm^2; must be > 0) over q = n, p."""
    p = dict(DEFAULT_PARAMS)
    p.update(params)
    n = np.asarray(n, float)
    x = np.asarray(_solve_xp(jnp.asarray(n), p))
    c1, c2 = _c12(n, p)
    em = lambda nq: p["h2m"] + c1 * n + c2 * nq
    return np.minimum(em((1 - x) * n), em(x * n))


class SkyrmeEOS(EOSModel):
    """Standard or extended (BSk) Skyrme EDF, npe-mu beta equilibrium."""

    name = "SkyrmeEOS"

    def __init__(self, n_min=1e-3, n_max=1.5, n_points=220,
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
            p.update(params)
        return p

    def composition(self, params: dict = None) -> Composition:
        """beta-equilibrium (x_p, x_e, x_mu) on ``self.n_grid``."""
        p = self._p(params)
        n = self.n_grid
        x_p = _solve_xp(n, p)
        _, n_e, n_mu, _ = lepton_sector(x_p * n, muons=True)
        return Composition(n=n, x_p=x_p, x_e=n_e / n, x_mu=n_mu / n)

    def landau_masses(self, params, comp):
        """(m*_n/m, m*_p/m) on ``comp`` (see :func:`landau_masses`)."""
        return landau_masses(self._p(params), comp.n, comp.x_p)

    def eos_table(self, params: dict = None):
        p = self._p(params)
        n = self._table_grid()
        x_p = _solve_xp(n, p)
        eps, pres = _eps_p(n, x_p, p)
        pres, ln_nu = monotone_core(n, pres, self.n_t)
        return finish_table(n, eps, pres, self.crust, self.n_t,
                            self.n_surface, self.gamma_crust, ln_nu,
                            self.allow_unstable_core)
