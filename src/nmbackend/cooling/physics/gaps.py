"""Pairing critical temperatures Tc(k_F), as NSCool's ``get_Tc`` builds them.

Fits (NSCool ``tc.f``; k_F in fm^-1, Tc in K):

| NSCool code | channel | source | form |
|---|---|---|---|
| SFN1S0 = 1 | n1S0 | SFB, Schwenk, Friman & Brown 2003 | spline of Delta(k_F), Tc = Delta / 1.76 |
| SFN1S0 = 2 | n1S0 | CCDK, Chen et al. 1993 | same |
| SFN3P2 = 101, 102, 103 | n3P2 | gaps "a", "b", "c" of Page et al. 2004 | Gaussian in k_F |
| SFP1S0 = 3 | p1S0 | T73, Takatsuka 1973 | spline of Tc(k_F), NSCool's y'' |
| SFP1S0 = 7 | p1S0 | CCDK | spline of Delta(k_F) |

Zhou+2026 Eq. 10 scalings, Delta(n) = s_y Delta_0(n / s_x), act as in the
nmbackend NSCool patch: s_y multiplies Tc (FN3P2, FP1S0); s_x evaluates the
gap at the k_F the star has at baryon density n / s_x (core zones only).
"""
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from ._spline import spline_y2, splint

__all__ = ["tc_n1s0", "tc_p1s0", "tc_n3p2_gauss", "N3P2_GAUSS", "kf_scaled",
           "StarTc", "star_tc"]

_MEV_K = 1.1604e10 / 1.76      # Delta [MeV] -> Tc [K], as in tc.f


def _f32(v):
    """Single-precision literals of NSCool DATA statements, widened."""
    return np.asarray(v, np.float32).astype(float)


class _Table(NamedTuple):
    k: np.ndarray
    t: np.ndarray
    y2: np.ndarray


# n1S0 SFB (tcn1_sfb): the spline is of Delta in MeV, scaled after.
_K = np.array([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0, 1.1, 1.175, 1.25, 1.3,
               1.35, 1.4, 1.45])
_D = np.array([0, 0, .09, .21, .35, .49, .61, .72, .79, .78, .70, .57, .44, .28,
               .19, .10, .03, 0])
_SFB = _Table(_K, _D, spline_y2(_K, _D, 0.0, 0.0))
# n1S0 CCDK (tcn1_ccdk): spline of Tc
_K = np.round(np.arange(1, 12) * 0.1, 2)
_D = np.array([0, .02, .14, .36, .60, .83, .86, .67, .35, .07, 0]) * _MEV_K
_CCDK_N = _Table(_K, _D, spline_y2(_K, _D, 0.0, 0.0))
# p1S0 CCDK (tcp1_ccdk)
_K = np.round(np.arange(0, 14) * 0.1, 2)
_D = np.array([0, .05, .19, .41, .66, .84, .97, 1.0, .90, .72, .49, .22, .07, 0]) * _MEV_K
_CCDK_P = _Table(_K, _D, spline_y2(_K, _D, 0.0, 0.0))
# p1S0 T73 (tcp1_t73): Tc and its y'' are tabulated (single precision)
_T73 = _Table(_f32([0.00, 0.30, 0.41, 0.48, 0.53, 0.58, 0.63, 0.67, 0.72, 0.76, 0.80, 0.84]),
              _f32([0.00e0, 7.58e8, 2.27e9, 3.03e9, 3.29e9, 3.00e9, 2.55e9, 2.11e9,
                    1.50e9, 8.45e8, 3.29e7, 0.00e0]),
              _f32([-2.49e+10, +1.00e+11, -6.79e+10, -5.58e+10, -3.16e+11, -1.03e+09,
                    -6.41e+10, -1.04e+10, -5.55e+10, -3.64e+11, +9.21e+11, -3.99e+11]))
del _K, _D


def _table_tc(tab: _Table, kf, scale=1.0):
    """Tc from a table; 0 at or outside its k_F range (as tc.f)."""
    inside = (kf > tab.k[0]) & (kf < tab.k[-1])
    k = jnp.clip(kf, tab.k[0], tab.k[-1])
    return jnp.where(inside, scale * splint(tab.k, tab.t, tab.y2, k), 0.0)


def tc_n1s0(code, kf):
    """Neutron 1S0 Tc (K) for NSCool code SFN1S0 (1 SFB, 2 CCDK), before the floor at 1 K."""
    if code == 1:
        return _table_tc(_SFB, kf, _MEV_K)
    if code == 2:
        return _table_tc(_CCDK_N, kf)
    raise NotImplementedError(f"SFN1S0 = {code}")


def tc_p1s0(code, kf):
    """Proton 1S0 Tc (K) for NSCool code SFP1S0 (3 T73, 7 CCDK), before the floor."""
    if code == 3:
        return _table_tc(_T73, kf)
    if code == 7:
        return _table_tc(_CCDK_P, kf)
    raise NotImplementedError(f"SFP1S0 = {code}")


#: SFN3P2 code -> (k_F of the maximum [fm^-1], width [fm^-1], Tc max [K])
N3P2_GAUSS = {101: (1.8, 0.5, 1.0e9), 102: (2.0, 0.5, 3.0e9), 103: (2.5, 0.7, 1.0e10)}


def tc_n3p2_gauss(kf, kf_max, dkf, tc_max):
    """Neutron 3P2 Tc of the minimal-cooling gaps (Page et al. 2004)."""
    return tc_max * jnp.exp(-(kf - kf_max) ** 2 / dkf ** 2)


def kf_scaled(sx, kf, nb, icore: int):
    """k_F at baryon density n / s_x in the core (NSCool patch, get_kf_scaled).

    Linear in n between core zones; k_F ~ n^(1/3) beyond the centre or the
    core boundary; crust zones (i > icore) unchanged.  ``nb`` decreases
    outwards from the centre (zone 0).
    """
    kf, nb = jnp.asarray(kf), jnp.asarray(nb)
    b = nb[:icore + 1] / sx
    xs, ys = nb[icore::-1], kf[icore::-1]            # ascending n
    mid = jnp.interp(b, xs, ys)
    hi = kf[0] * (b / nb[0]) ** (1.0 / 3.0)
    lo = kf[icore] * (b / nb[icore]) ** (1.0 / 3.0)
    core = jnp.where(b >= nb[0], hi, jnp.where(b <= nb[icore], lo, mid))
    return jnp.concatenate([core, kf[icore + 1:]])


class StarTc(NamedTuple):
    tcn: jnp.ndarray   # neutron Tc per zone (K); 1 K means unpaired
    tcp: jnp.ndarray   # proton Tc per zone (K)
    isf: jnp.ndarray   # last zone of n3P2 pairing, NSCool's isf (-1 if none)


def star_tc(gaps, kfn, kfp, nb, icore: int, idrip: int, *, sy_n3p2=None,
            sx_n3p2=None, sy_p1s0=None, sx_p1s0=None, n3p2_params=None) -> StarTc:
    """NSCool ``get_Tc`` on a star's zones (index 0 = centre).

    ``gaps`` is a ``GapModel`` (only its integer codes are used here unless
    the keyword overrides are None).  The scalings and the 3P2 Gaussian
    parameters can be passed as JAX values to differentiate through them.
    ``isf`` follows get_Tc for codes 101-103: the run of zones from the
    centre in which the 3P2 Tc is at least the 1S0 Tc.
    """
    sy_n3p2 = gaps.sy_n3p2 if sy_n3p2 is None else sy_n3p2
    sx_n3p2 = gaps.sx_n3p2 if sx_n3p2 is None else sx_n3p2
    sy_p1s0 = gaps.sy_p1s0 if sy_p1s0 is None else sy_p1s0
    sx_p1s0 = gaps.sx_p1s0 if sx_p1s0 is None else sx_p1s0
    kfn, kfp, nb = jnp.asarray(kfn), jnp.asarray(kfp), jnp.asarray(nb)
    i = jnp.arange(kfn.shape[0])
    one = jnp.ones_like(kfn)
    # n1S0 (FN1S0 = 1 in GapModel)
    tcn = one
    if gaps.n1s0 != 0:
        tcn = jnp.where(i <= idrip, jnp.maximum(1.0, tc_n1s0(int(gaps.n1s0), kfn)), 1.0)
    # n3P2
    isf = jnp.asarray(-1)
    if gaps.n3p2 != 0:
        code = int(gaps.n3p2)
        if code not in N3P2_GAUSS:
            raise NotImplementedError(f"SFN3P2 = {code}")
        par = N3P2_GAUSS[code] if n3p2_params is None else n3p2_params
        t3 = tc_n3p2_gauss(kf_scaled(sx_n3p2, kfn, nb, icore), *par) * sy_n3p2
        win = (t3 >= tcn) & (i <= idrip)
        tcn = jnp.where(win, t3, tcn)
        isf = jnp.sum(jnp.cumprod(win.astype(int))) - 1
    # p1S0
    tcp = one
    if gaps.p1s0 != 0:
        kps = kf_scaled(sx_p1s0, kfp, nb, icore)
        tcp = jnp.where(i <= icore,
                        jnp.maximum(1.0, tc_p1s0(int(gaps.p1s0), kps)) * sy_p1s0, 1.0)
    return StarTc(jnp.abs(tcn), jnp.abs(tcp), isf)
