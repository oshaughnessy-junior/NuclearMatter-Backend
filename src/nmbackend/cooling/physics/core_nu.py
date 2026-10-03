"""Core neutrino emissivities (erg cm^-3 s^-1), as NSCool's ``neutrino_core.f``.

All functions are elementwise over zones.  T is the local temperature (K),
k_F in fm^-1, m* the Landau effective mass over the bare mass, nb in fm^-3.
``i`` is the zone index (0 = centre) and ``isf`` NSCool's last n3P2 zone;
NSCool treats neutron pairing as 3P2 for i < isf in the Urca and
bremsstrahlung routines and for i <= isf in pair breaking (zone isf
itself differs between them; we keep both).

| Function | NSCool routine | Source cited there |
|---|---|---|
| ``murca`` | numurca_nucl | Yakovlev & Levenfish 1995 |
| ``brem`` | nubrem_nucl | same |
| ``durca`` | nudurca_h (n-p channel) | Levenfish & Yakovlev 1994; Lattimer+1991 |
| ``pbf_1s0``, ``pbf_n3p2`` | nu_1s0_pbf, nu_n3p2_B_pbf | Yakovlev+1999; Kaminker+1999; Leinson 2010 |
"""
from __future__ import annotations

import functools
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from ._spline import natural_y2_matrix, spline_y2, splint2
from .superfluid import below, f32, fexp, u_1s0, u_3p2b

__all__ = ["murca", "brem", "durca", "durca_flag", "r_durca", "pbf_1s0",
           "pbf_n3p2", "pbf_neutron", "sf_suppression_table"]

_PI = 3.14159265


def _sq(a, b):
    return jnp.sqrt(a ** 2 + b ** 2)


# ---- modified Urca --------------------------------------------------------
def _r_mu_a(u):     # rmurca_n_p1s0 = rmurca_p_n1s0
    return (fexp(3.4370 - _sq(3.4370, u)) * 0.5
            * ((0.1477 + _sq(0.8523, 0.1175 * u)) ** 7.5
               + (0.1477 + _sq(0.8523, 0.1297 * u)) ** 5.5))


def _r_mu_b(u):     # rmurca_n_n1s0 = rmurca_p_p1s0
    return fexp(5.3390 - _sq(5.3390, 2.0 * u)) * (0.2414 + _sq(0.7586, 0.1318 * u)) ** 7.0


def _r_mu_p_3p2(u):
    return (fexp(2.3980 - _sq(2.3980, u)) * 0.5
            * ((0.1612 + _sq(0.8388, 0.1117 * u)) ** 7
               + (0.1612 + _sq(0.8388, 0.1274 * u)) ** 5))


def murca(T, kfe, kfm, kfn, kfp, mstn, mstp, tcn, tcp, i, isf):
    """Modified Urca, neutron plus proton branch (numurca_nucl)."""
    t9 = T / 1e9
    on = kfn * kfp != 0.0
    kfn_ = jnp.where(on, kfn, 1.0)
    kfp_ = jnp.where(on, kfp, 1.0)
    kfe_ = jnp.where(kfe != 0.0, kfe, 1.0)
    ab = (1.76 - 0.63 * (1.68 / kfn_) ** 2) * 0.68
    qn = 8.55e21 * mstn ** 3 * mstp * (kfe / 1.68 + kfm / 1.68) * ab * t9 ** 8
    qp = (8.55e21 * mstn * mstp ** 3 * (kfe / 1.68)
          * (kfe + 3.0 * kfp_ - kfn_) ** 2 / (8.0 * kfe_ * kfp_) * ab * t9 ** 8)
    # neutron pairing: 1S0 for i >= isf, 3P2 below
    mn, tn = below(T, tcn)
    trip = i < isf
    u1, u3 = u_1s0(tn), u_3p2b(tn)
    r_nn = jnp.where(trip, 39.1 * tn * fexp(-1.188 / tn) * _r_mu_p_3p2(u3), _r_mu_b(u1))
    r_pn = jnp.where(trip, _r_mu_p_3p2(u3), _r_mu_a(u1))
    r_nn, r_pn = jnp.where(mn, r_nn, 1.0), jnp.where(mn, r_pn, 1.0)
    mp, tp = below(T, tcp)
    up = u_1s0(tp)
    r_np = jnp.where(mp, _r_mu_a(up), 1.0)
    r_pp = jnp.where(mp, _r_mu_b(up), 1.0)
    q = jnp.minimum(r_np, r_nn) * qn + jnp.minimum(r_pp, r_pn) * qp
    return jnp.where(on, q, 0.0)


# ---- nucleon-nucleon bremsstrahlung -------------------------------------
def _r_nn(u):       # rbrem_nn_n1s0 (also pp for p1S0, nn for n3P2)
    return ((0.1747 + _sq(0.8253, 0.07933 * u)) ** 2 * fexp(4.228 - _sq(4.228, 4.0 * u)) / 2.0
            + (0.7333 + _sq(0.2667, 0.1678 * u)) ** 7.5 * fexp(7.762 - _sq(7.762, 9.0 * u)) / 2.0)


def _r_np(u):       # rbrem_np_p1s0 = rbrem_np_n1s0 = rbrem_np_n3p2B
    return ((0.9982 + _sq(0.0018, 0.3815 * u)) * fexp(1.306 - _sq(1.306, u)) / 2.732
            + (0.3949 + _sq(0.6051, 0.2666 * u)) ** 7 * fexp(3.303 - _sq(3.303, 4.0 * u)) / 1.577)


def brem(T, kfn, kfp, mstn, mstp, tcn, tcp, i, isf):
    """nn + np + pp bremsstrahlung in the core (nubrem_nucl), three flavours."""
    c = 3.0 * (T / 1e9) ** 8
    qnn = c * 7.4e19 * mstn ** 4 * (kfn / 1.68) * 0.59 * 0.56
    qnp = c * 1.5e20 * mstn ** 2 * mstp ** 2 * (kfp / 1.68) * 1.06 * 0.66
    qpp = c * 7.4e19 * mstp ** 4 * (kfp / 1.68) * 0.11 * 0.7
    mn, tn = below(T, tcn)
    u = jnp.where(i < isf, u_3p2b(tn), u_1s0(tn))
    rnn_n = jnp.where(mn, _r_nn(u), 1.0)
    rnp_n = jnp.where(mn, _r_np(u), 1.0)
    mp, tp = below(T, tcp)
    up = u_1s0(tp)
    rnp_p = jnp.where(mp, _r_np(up), 1.0)
    rpp_p = jnp.where(mp, _r_nn(up), 1.0)
    return rnn_n * qnn + jnp.minimum(rnp_p, rnp_n) * qnp + rpp_p * qpp


# ---- direct Urca ---------------------------------------------------------
def durca_flag(kfe, kfm, kfn, kfp):
    """NSCool idurca_np: 0 closed, 1 electron triangle, 2 electron and muon."""
    kfe, kfm, kfn, kfp = (jnp.asarray(x) for x in (kfe, kfm, kfn, kfp))
    tri = lambda a, b, c: (a < b + c) & (b < a + c) & (c < a + b)
    e = tri(kfp, kfn, kfe)
    mu = e & tri(kfp, kfn, kfm)
    return e.astype(int) + mu.astype(int)


def _r1(u):
    return (0.2312 + _sq(0.7688, 0.1438 * u)) ** 5.5 * fexp(3.427 - _sq(3.427, u))


def _r3(u):
    return (0.2546 + _sq(0.7454, 0.1284 * u)) ** 5 * fexp(2.701 - _sq(2.701, u))


def _r_1s0_1s0(v1, v2):
    """func_r_1s0_1s0 (Levenfish & Yakovlev 1994, both 1S0)."""
    gamma = 5040.0 / 457.0 / _PI ** 6
    u = v1 ** 2 + v2 ** 2
    w = v1 ** 2 - v2 ** 2
    u1 = 1.8091 + jnp.sqrt(v1 ** 2 + 2.2476 ** 2)
    u2 = 1.8091 + jnp.sqrt(v2 ** 2 + 2.2476 ** 2)
    root = jnp.sqrt(w ** 2 + 16.350 * u + 45.171)
    p = (u + 12.421 + root) / 2.0
    q = (u + 12.421 - root) / 2.0
    ps = (u + jnp.sqrt(w ** 2 + 5524.8 * u + 6.7737)) / 2.0
    pe = (u + 0.43847 + jnp.sqrt(w ** 2 + 8.3680 * u + 491.32)) / 2.0
    d = ((u1 * u2) ** 1.5 / (2.0 * 4.0567 ** 5) * (u1 ** 2 + u2 ** 2)
         * jnp.exp(-u1 - u2 + 8.1134))
    lg = jnp.log((jnp.sqrt(p) + jnp.sqrt(p - q)) / jnp.sqrt(q))
    k0 = (jnp.sqrt(p - q) / 120.0 * (6.0 * p ** 2 + 83.0 * p * q + 16.0 * q ** 2)
          - jnp.sqrt(p) * q / 8.0 * (4.0 * p + 3.0 * q) * lg)
    k1 = (_PI ** 2 * jnp.sqrt(p - q) / 6.0 * (p + 2.0 * q)
          - _PI ** 2 / 2.0 * q * jnp.sqrt(p) * lg)
    k2 = 7.0 * _PI ** 4 / 60.0 * jnp.sqrt(p - q)
    s = (gamma * (k0 + k1 + 0.42232 * k2) * jnp.sqrt(_PI / 2.0) * ps ** 0.25
         * jnp.exp(-jnp.sqrt(pe)))
    return u / (u + 0.9163) * s + d


@functools.lru_cache(maxsize=None)
def sf_suppression_table(path=None):
    """NSCool's Code/Data_Files/SF_suppression.dat as spline arrays.

    log10 R(log10 tau_p, log10 tau_n) for dUrca with p1S0 and n3P2 pairing.
    NSCool has no licence, so the table is read from the NSCool build
    (``build_nscool.sh``), not shipped with nmbackend.  ``durca`` needs it
    even for stars without dUrca; the default path is fixed at first use.
    """
    if path is None:
        from ..runner import default_home
        path = default_home() / "Code" / "Data_Files" / "SF_suppression.dat"
    v = np.array(Path(path).read_text().split(), float)
    if v.size != 35 + 35 + 35 * 35:
        raise ValueError(f"{path}: expected 35 + 35 + 35x35 numbers")
    x1, x2 = v[:35][::-1], v[35:70][::-1]                 # read backwards
    lgr = v[70:].reshape(35, 35)[::-1, ::-1]
    y2 = np.stack([spline_y2(x2, row) for row in lgr])
    return x1, x2, lgr, y2, natural_y2_matrix(x1)


def _r_1s0_3p2b(t1, t2, table):
    """func_r_1s0_3P2B: spline in (log10 T/Tcp, log10 T/Tcn); beyond radius 3
    NSCool evaluates on the unit circle and multiplies by exp(-r/3)."""
    l1, l2 = jnp.log10(t1), jnp.log10(t2)
    lt = jnp.sqrt(l1 ** 2 + l2 ** 2)
    far = lt > 3.0
    s = jnp.where(far, lt, 1.0)
    ev = jnp.vectorize(lambda a, b: splint2(*table[:4], table[4], a, b))
    return 10.0 ** ev(l1 / s, l2 / s) * jnp.where(far, jnp.exp(-lt / 3.0), 1.0)


def r_durca(T, tcn, tcp, i, isf, table=None):
    """Pairing suppression of n-p dUrca (nudurca_h)."""
    table = sf_suppression_table() if table is None else table
    sn = T <= tcn
    sp = T <= tcp
    _, tn = below(T, tcn, strict=False)
    _, tp = below(T, tcp, strict=False)
    trip = i < isf
    vp, vn1, vn3 = u_1s0(tp), u_1s0(tn), u_3p2b(tn)
    r_p = _r1(vp)
    r_n = jnp.where(trip, _r3(vn3), _r1(vn1))
    r_both = jnp.where(trip, _r_1s0_3p2b(tp, tn, table), _r_1s0_1s0(vp, vn1))
    return jnp.where(sn & sp, r_both, jnp.where(sn, r_n, jnp.where(sp, r_p, 1.0)))


def durca(T, kfe, kfm, kfn, kfp, mstn, mstp, nb, Ye, Ymu, tcn, tcp, i, isf, table=None):
    """n-p direct Urca with electrons, plus muons where their triangle closes."""
    flag = durca_flag(kfe, kfm, kfn, kfp)
    mu = flag == 2
    ymu = jnp.where(mu, jnp.abs(Ymu), 1.0)        # Ymu = 0: keep d/dYmu finite
    lep = ((jnp.abs(Ye) * nb / 0.16) ** (1.0 / 3.0)
           + jnp.where(mu, (ymu * nb / 0.16) ** (1.0 / 3.0), 0.0))
    q = 4.24e27 * mstn * mstp * (T / 1e9) ** 6 * lep
    return jnp.where(flag > 0, q * r_durca(T, tcn, tcp, i, isf, table), 0.0)


# ---- pair breaking and formation --------------------------------------------
_C011 = f32(0.01130)
_C3P2 = (f32(0.3191), f32(0.3511), f32(0.7591))


def _ctrl_1s0(v):
    x = 0.602 * v ** 2 + 0.5942 * v ** 4 + 0.288 * v ** 6
    y = jnp.sqrt(0.5547 + jnp.sqrt(0.4453 ** 2 + _C011 * v ** 2))
    return x * y * jnp.exp(-jnp.sqrt(4.0 * v ** 2 + 2.245 ** 2) + 2.245)


def _ctrl_3p2b(v):
    a, b, c = _C3P2
    x = (1.204 * v ** 2 + 3.733 * v ** 4 + a * v ** 6) / (1.0 + b * v ** 2)
    y = (c + jnp.sqrt(0.2409 ** 2 + 0.3145 * v ** 2)) ** 2
    return x * y * jnp.exp(-jnp.sqrt(4.0 * v ** 2 + 0.4616 ** 2) + 0.4616)


def _pbf(T, kf, mst, a, ctrl, m, v):
    vf = kf * 197.0 / mst / 940.0
    q = 1.170e21 * mst ** 2 * vf * (T / 1e9) ** 7 * 3.0 * a * ctrl(v)
    return jnp.where(m, q, 0.0)


def pbf_1s0(T, tc, mst, kf):
    """Singlet pair breaking, axial part only (nu_1s0_pbf)."""
    m, tau = below(T, tc, strict=False)
    m, mst = m & (mst > 0), jnp.where(mst > 0, mst, 1.0)   # no neutrons: m* = 0
    vf = kf * 197.0 / mst / 940.0
    a = 1.60 * vf ** 2 * (mst ** 2 + 11.0 / 42.0)
    return _pbf(T, kf, mst, a, _ctrl_1s0, m, u_1s0(tau))


def pbf_n3p2(T, tc, mst, kf, ca=0.5):
    """n3P2 (m=0) pair breaking with a_A = c_A g_A^2 (nu_n3p2_B_pbf, nmbackend patch).

    NSCool's gap here is sqrt(1-tau)(0.7893 + 1.764/tau); the Urca routines
    use 1.188/tau (Yakovlev+1999 Table 1) for the same gap.
    """
    m, tau = below(T, tc, strict=False)
    m, mst = m & (mst > 0), jnp.where(mst > 0, mst, 1.0)
    v = jnp.sqrt(1.0 - tau) * (0.7893 + 1.764 / tau)
    return _pbf(T, kf, mst, ca * 1.26 ** 2, _ctrl_3p2b, m, v)


def pbf_neutron(T, tcn, mstn, kfn, i, isf, ca=0.5):
    """Neutron pair breaking as neutrino.f dispatches it: (n1S0 for i > isf,
    n3P2 for i <= isf).  Applies in the crust as well as the core."""
    q1 = jnp.where(i > isf, pbf_1s0(T, tcn, mstn, kfn), 0.0)
    q3 = jnp.where(i <= isf, pbf_n3p2(T, tcn, mstn, kfn, ca), 0.0)
    return q1, q3
