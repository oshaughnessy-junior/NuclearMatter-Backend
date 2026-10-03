"""Crust thermal conductivity (erg cm^-1 s^-1 K^-1), NSCool ICON_CRUST = 3.

``con_crust`` with ICON_CRUST = 3 (``conductivity_crust.f``), for
rho >= 6e7 g/cm^3:

| Piece | NSCool routine | Source cited there |
|---|---|---|
| electron-phonon (solid) and electron-ion (liquid) | con_e_phon_ion_GYP, OYAFORM | Gnedin, Yakovlev & Potekhin 2001 App.; Potekhin+1999 |
| electron-impurity (solid only) | con_crust_e_imp_YU | Yakovlev & Urpin 1980 |
| electron-electron | con_crust_ee | Shternin & Yakovlev 2006 |

The GYP piece takes the nuclear composition from the smooth-composition
model of OYAFORM (Oyamatsu), not from the zone's A and Z; the solid test
(Gamma > 210) and the impurity term use the zone's a_cell and Z.  Most
literals of OYAFORM are single precision and are kept so.
"""
from __future__ import annotations

import jax.numpy as jnp

from .superfluid import f32

__all__ = ["oyaform", "lambda_gyp", "lambda_imp", "lambda_ee", "lambda_crust"]

_F = f32
_TH = _F(.333333)


def _soyam(t, x):
    return (x ** 3 - 9.0 * x ** (3.0 + t) / (3.0 + t)
            + 9.0 * x ** (3.0 + 2.0 * t) / (3.0 + 2.0 * t) - x ** (3.0 + 3.0 * t) / (1.0 + t))


def oyaform(bard, after_drip):
    """Smooth composition (OYAFORM): Z, A_nuc, A_cell, xnuc, xnuct at baryon
    density ``bard`` (fm^-3); ``after_drip`` selects NSCool's Index 3 over 30."""
    pi = 3.14159265
    v = pi / _F(.75)
    # Index 30: below neutron drip
    f = jnp.log(1.0 + bard / 5.0e-9)
    rp = _F(5.688) + _F(0.02628) * f + _F(0.009468) * f * f
    rn = _F(5.788) + _F(0.02077) * f + _F(0.01489) * f * f
    np_in = _F(0.0738) + _F(1.22e-4) * f - _F(1.641e-4) * f * f
    nn_in = _F(0.0808) + _F(1.688e-4) * f + _F(9.439e-5) * f * f
    tp30 = 6.0
    nin = v * rn ** 3 * nn_in * _soyam(tp30, 1.0)
    z30 = v * rp ** 3 * np_in * _soyam(tp30, 1.0)
    a30 = z30 + nin
    rws30 = (a30 * _F(.75) / pi / bard) ** _TH
    # Index 3: spheres after drip
    g = bard * 100.0
    f = jnp.log(g)
    rws = _F(31.68) - _F(8.400) * f - _F(0.2380) * f * f + _F(0.1152) * f ** 3
    tn = 1.0 / (_F(0.2027) + _F(0.004506) * g)
    rn3 = _F(9.406) + _F(1.481) * f + _F(0.4625) * f * f + _F(0.05738) * f ** 3
    dn_n = (_F(9.761) - _F(1.322) * f - _F(0.5544) * f * f - _F(0.07624) * f ** 3) / 100.0
    nin3 = v * rn3 ** 3 * dn_n * _soyam(tn, jnp.minimum(1.0, rws / rn3))
    tp3 = 1.0 / (_F(0.1558) + _F(2.225e-3) * g + _F(9.452e-4) * g * g)
    rp3 = _F(8.345) + _F(0.7767) * f + _F(0.1333) * f * f + _F(0.008707) * f ** 3
    np3 = (_F(4.040) - _F(1.097) * f - _F(0.0723) * f * f + _F(0.0225) * f ** 3) / 100.0
    z3 = v * rp3 ** 3 * np3 * _soyam(tp3, jnp.minimum(1.0, rws / rp3))
    nfree = bard * v * rws ** 3 - z3 - nin3
    a3 = z3 + nfree + nin3
    anuc3 = jnp.where(rn3 > rws, a3, z3 + nin3 + nfree * (rn3 / rws) ** 3)
    w = lambda a, b: jnp.where(after_drip, a, b)
    z, anuc, a = w(z3, z30), w(anuc3, a30), w(a3, a30)
    tp, rp, npi, rws = w(tp3, tp30), w(rp3, rp), w(np3, np_in), w(rws, rws30)
    rp2 = rp * jnp.sqrt((1.0 - 15.0 / (5.0 + tp) + 15.0 / (5.0 + 2.0 * tp)
                         - 5.0 / (5.0 + 3.0 * tp)) / _soyam(tp, 1.0))
    xnuc = rp2 / rws
    return z, anuc, a, xnuc, xnuc * tp / (_F(.6) + tp)


def _e1(x):
    """exp_int: E1(x), NSCool's rational / series fit."""
    xs = jnp.maximum(x, 1.0)
    num = xs ** 4 + 8.5733287401 * xs ** 3 + 18.0590169730 * xs ** 2 + 8.6347608925 * xs + 0.2677737343
    den = xs ** 4 + 9.5733223454 * xs ** 3 + 25.6329561486 * xs ** 2 + 21.0996530827 * xs + 3.9584969228
    big = num / den / xs * jnp.exp(-xs)      # NSCool: / (x exp(x)); no overflow
    xl = jnp.clip(x, 1e-300, 1.0)
    small = (-0.57721566 + 0.99999193 * xl - 0.24991055 * xl ** 2 + 0.05519968 * xl ** 3
             - 0.00976004 * xl ** 4 + 0.00107857 * xl ** 5 - jnp.log(xl))
    return jnp.where(x >= 1.0, big, small)


def _get_lam(s, w):
    eps = 0.05
    l = jnp.log((s + 1.0) / s)
    ws = jnp.where(w > 0, w, 1.0)
    a1 = 0.5 * (_e1(ws) + jnp.log(ws) + 0.5772156)
    a2 = (jnp.exp(-ws) - 1.0 + ws) / (2.0 * ws)
    b1 = w * ((2.0 * s + 1.0) / (2.0 * s + 2.0) - s * l)
    b2 = w * ((1.0 - 3.0 * s - 6.0 * s ** 2) / (4.0 * s + 4.0) + 1.5 * l)
    c1 = 0.5 * (l - 1.0 / (s + 1.0))
    c2 = (2.0 * s + 1.0) / (2.0 * s + 2.0) - s * l
    c_a = (s <= eps) & (s * w <= eps)
    c_b = w <= eps
    c_c = w > 1.0 / eps
    # the general branch only for w <= 1/eps: keep its exp(s w) finite elsewhere
    ws = jnp.where(c_c, 1.0, ws)
    sw = s * ws
    de = jnp.exp(sw) * (_e1(sw) - _e1(sw + ws))
    d1 = 0.5 * (l + s / (s + 1.0) * (1.0 - jnp.exp(-ws)) - (1.0 + sw) * de)
    d2 = 0.5 * ((jnp.exp(-ws) - 1.0 + ws) / ws - s ** 2 / (s + 1.0) * (1.0 - jnp.exp(-ws))
                - 2.0 * s * l + s * (2.0 + sw) * de)
    pick = lambda a, b, c, d: jnp.where(c_a, a, jnp.where(c_b, b, jnp.where(c_c, c, d)))
    return pick(a1, b1, c1, d1), pick(a2, b2, c2, d2)


def lambda_gyp(T, rho, rhodrip):
    """Electron-phonon (solid) or electron-ion (liquid) conductivity, GYP."""
    pi, c, kb, hb, a_f = 3.14159265, 2.99e10, 1.380e-16, 1.054e-27, 1.0 / 137.0
    e, me, mu = 4.803e-10, 9.109e-28, 1.66e-24
    u_1, u_2 = 2.80, 13.00
    third = _F(1.0 / 3.0)
    z, a1, a, xnuc, xnuct = oyaform(rho / mu * 1e-39, rho > rhodrip)
    n_i = rho / (a * mu)
    n_e = z * n_i
    kf = (3.0 * pi ** 2 * n_e) ** third
    pf = hb * kf
    m_st = jnp.sqrt(me ** 2 + (pf / c) ** 2)
    ef = m_st * c ** 2
    vf = pf / m_st
    x = pf / me / c
    t_p = hb * jnp.sqrt(4.0 * pi * e ** 2 * z ** 2 * n_i / a1 / mu) / kb
    tp = T / t_p
    bz = pi * a_f * z * vf / c
    ai = (3.0 / (4.0 * pi * n_i)) ** third
    gam = z ** 2 * e ** 2 / (kb * T * ai)
    s_d = 1.0 / (2.0 * kf * ai / jnp.sqrt(3.0 * gam)) ** 2
    s_i = s_d * (1.0 + 0.06 * gam) * jnp.exp(-jnp.sqrt(gam))
    s = (s_i + a_f / pi * c / vf) * jnp.exp(-bz)
    w = (u_2 / s_d) * (1.0 + bz / 3.0)
    w1 = 14.73 * xnuc ** 2 * (1.0 + z * jnp.sqrt(xnuc) / 13.0) * (1.0 + bz / 3.0)
    g_s = 1.0 / jnp.sqrt(1.0 + 0.0361 / z ** third / tp ** 2) * (1.0 + 0.122 * bz ** 2)
    g_l = g_s + (0.0105 * tp / (tp ** 2 + 0.0081) ** 1.5 * (1.0 + (vf / c) ** 3 * bz)
                 * (1.0 - 1.0 / z) * (1.0 + xnuct ** 2 * jnp.sqrt(2.0 * z)))
    d = jnp.exp(-0.42 * jnp.sqrt(x / a / z) * u_1 * jnp.exp(-9.1 * tp))
    l1a, l2a = _get_lam(s, w + w1)
    l1b, l2b = _get_lam(s, w1)
    lam = (l1a - (vf / c) ** 2 * l2a) - (l1b - (vf / c) ** 2 * l2b)
    t_u = t_p * z ** third * a_f / 3.0 / vf * c
    lam0 = 50.0 * x ** 0.5 / a1 ** 0.5 / z
    ww = jnp.exp(-t_u / T)
    lam_s = lam * g_s * d * ww + lam0 * _F(4.0 / 3.0) * a_f * c / vf * tp ** 5 * (1.0 - ww)
    lam_l = lam * g_l * d * ww + lam0 * tp ** 3 * (1.0 - ww)
    nu0 = 4.0 * z * ef * a_f ** 2 / 3.0 / pi / hb
    del lam_s                                  # electrical conductivity only
    return pi ** 2 * kb ** 2 * T * n_e / (3.0 * m_st * nu0 * lam_l)


def lambda_imp(T, rho, A, Z, qimp):
    """Electron-impurity conductivity (con_crust_e_imp_YU); A = a_cell."""
    x = 1.00884 * (Z / A * rho / 1e6) ** _F(1.0 / 3.0)
    beta = x / jnp.sqrt(1.0 + x ** 2)
    clog = jnp.log(1.0 / (0.048196 / jnp.sqrt(beta))) - 0.5 * (1.0 + beta ** 2)
    nu = 1.75e16 * jnp.sqrt(1.0 + x ** 2) * qimp / Z * clog
    return 4.04e17 * x ** 3 / jnp.sqrt(1.0 + x ** 2) * (T / 1e8) * (1e16 / nu)


def lambda_ee(T, kfe):
    """Electron-electron conductivity (con_crust_ee, Shternin & Yakovlev 2006)."""
    pi = jnp.pi
    kb, me, e, hbar, cl = 1.380e-16, 9.109e-28, 4.803206e-10, 1.054572e-27, 2.99792458e10
    ne = kfe ** 3 / 3.0 / pi ** 2 * 1e39
    xe = 197.3 * kfe / 0.511
    ge = jnp.sqrt(1.0 + xe ** 2)
    u = xe / ge
    om = jnp.sqrt(4.0 * pi * e ** 2 * ne / (ge * me))
    th = jnp.sqrt(3.0) * hbar * om / kb / T
    i_l = ((0.1587 - 0.02538 / (1.0 + 0.0435 * th))
           * jnp.log(1.0 + 128.56 / (37.1 * th + 10.83 * th ** 2 + th ** 3)) / u)
    a = 20.0 + 450.0 * u ** 3
    c1, c2 = 0.05067 + 0.03216 * u ** 2, 0.0254 + 0.04127 * u ** 4
    c = a * jnp.exp(c1 / c2)
    i_t = (u ** 3 * (2.404 / c + (c2 - 2.404 / c) / (1.0 + 0.1 * th * u))
           * jnp.log(1.0 + c / (a * th * u + th ** 2 * u ** 2)))
    a = 12.2 + 25.2 * u ** 3
    b = 1.0 - 0.75 * u
    c1, c2 = 0.123636 + 0.016234 * u ** 2, 0.0762 + 0.05714 * u ** 4
    c = a * jnp.exp(c1 / c2)
    i_lt = (u * (18.52 * u ** 2 / c + (c2 - 18.52 * u ** 2 / c) / (1.0 + 0.1558 * th ** b))
            * jnp.log(1.0 + c / (a * th + 10.83 * th ** 2 * u ** 2 + (th * u) ** _F(8.0 / 3.0))))
    alpha = 1.0 / 137.036
    return pi ** 3 * kb ** 3 * T ** 2 / (108.0 * alpha ** 2 * hbar ** 2 * cl * (i_l + i_t + i_lt))


def lambda_crust(T, rho, A, Z, kfe, qimp, rhodrip):
    """con_crust, ICON_CRUST = 3, rho >= 6e7 g/cm^3.  A = a_cell, Z per nucleus."""
    gamma = _F(2.273e5) * Z ** 2 * (rho / A) ** _F(1.0 / 3.0) / T
    lam = lambda_gyp(T, rho, rhodrip)
    li = lambda_imp(T, rho, A, Z, qimp)
    solid = 1.0 / (1.0 / lam + 1.0 / li)
    lam = jnp.where(gamma > 210.0, solid, lam)
    lee = lambda_ee(T, kfe)
    return lam * lee / (lam + lee)
