"""NSCool ICON_CORE=2 lepton thermal conductivity, including proton pairing.

The Shternin–Yakovlev 2007 prescription used by NSCool (2009 routine).
NSCool deliberately excludes its computed baryon conductivity from this
return value. This is a reference prescription, not a modern total-core model;
its superconducting transverse screening uses the Pippard-limit fit.
"""
from __future__ import annotations

import jax.numpy as jnp

from .superfluid import below, f32, u_1s0

__all__ = ["lambda_core"]


def lambda_core(T, kfe, kfm, kfp, mstp, tcp):
    """Thermal conductivity [erg cm^-1 s^-1 K^-1] of core electrons and muons.

    Mirrors NSCool ``con_core_lep`` and the lepton-only ``con_core`` sum.
    T is local [K], kF [fm^-1], mstp the proton Landau/bare mass ratio,
    tcp [K]. Inputs broadcast; T > 0, kF >= 0, mstp > 0 where protons exist.
    Absent leptons contribute zero. Derivatives are intended away from
    species onsets and the pairing transition (the shared gap helper clips Tc).
    Hyperon scattering, baryon conduction and magnetic anisotropy are excluded.
    """
    T, kfe, kfm, kfp, mstp, tcp = map(jnp.asarray, (T, kfe, kfm, kfp, mstp, tcp))
    e = jnp.where(kfe > 0, kfe, 1.0)
    has_mu = kfm > 0
    m = kfm/e
    ms = jnp.where(has_mu, m, 1.0)
    p = kfp/e
    mp = jnp.where(kfp > 0, mstp, 0.0)
    r0, t8 = 1.68/e, T/1e8
    ql3 = (0.00929*(1 + m + 2.83*mp*r0*p))**-1.5
    qt2 = 1/(0.00929*(1 + m**2 + p**2))
    ee_par = 1.43e11*r0*ql3*t8**2
    ep_par = 1.15e12*mp**2*r0**2*ql3*t8**2
    ee_per = 6.49e14*qt2*t8
    ee_pri = 4.38e12*r0**f32(2/3)*qt2**f32(1/3)*ql3**f32(2/3)*t8**f32(5/3)

    paired, tau = below(T, tcp, strict=False)
    paired = paired & (kfp > 0)
    y = u_1s0(jnp.where(paired, tau, 0.5))
    # Preserve the reference value exactly at Tc without evaluating sqrt(0)
    # in an AD-active branch. Transition derivatives are outside the contract.
    y = jnp.where(T == tcp, 0.0, y)
    sqrt_y = jnp.where(y > 0, jnp.sqrt(jnp.where(y > 0, y, 1.0)), 0.0)
    ps = jnp.where(kfp > 0, kfp, e)
    r = (e**2+kfm**2)/ps**2
    rl = (r+1)**f32(1/3)/((r+1)**2-0.757*y+(0.50651*y)**2)**f32(1/6)
    p1 = 0.48-0.17*r
    p3 = ((1-p1)*54/4/3.1415926535**2/r)**2
    rt = p1*jnp.exp(-0.14*y**2)+(1-p1)/jnp.sqrt(1+p3*y**2)
    rp = (0.998+(2.04+0.68*sqrt_y+5.7*y**2+1.71*y**4)*jnp.exp(-1.04*y))*jnp.exp(-jnp.sqrt(1.23+y**2))
    rl, rt, rp = (jnp.where(paired, x, 1.0) for x in (rl, rt, rp))

    nu_e = (ee_par*(1+has_mu)+ep_par*rp
            + ee_per*(1+m**2+p**2)*rt + ee_pri*rl)
    nu_m = jnp.where(has_mu,
        (2*ee_par+ep_par*rp)/ms + ee_per*(m**3+m+m*p**2)*rt + ee_pri*m**3*rl, 0.0)
    em = jnp.where(has_mu, ee_pri*m**2*rl, 0.0)
    me = jnp.where(has_mu, ee_pri/ms*rl, 0.0)
    denominator = jnp.where(has_mu, nu_e*nu_m-em*me, 1.0)
    tau_e = jnp.where(has_mu, (nu_m-em)/denominator, 1/nu_e)
    tau_m = jnp.where(has_mu, (nu_e-me)/denominator, 0.0)
    lam = 1.70e24*t8*1e15*(tau_e*(e/1.68)**2+tau_m*(kfm/1.68)**2)
    return jnp.where(kfe > 0, lam, 0.0)
