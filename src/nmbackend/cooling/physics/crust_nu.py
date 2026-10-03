"""Crust neutrino emissivities (erg cm^-3 s^-1), as NSCool's ``neutrino_crust.f``.

| Function | NSCool routine | Source cited there |
|---|---|---|
| ``eebrem`` | neebrem | Jaikumar, Gale & Page 2005 (table Nalpha.dat) |
| ``npb`` | npb_new | Kaminker+1999, Eq. 40 |
| ``plasma`` | nplasma | Haft, Raffelt & Weiss 1994; Itoh+1996 |
| ``brem_nn_crust`` | nubrem_crust_nn | Yakovlev & Levenfish 1995 |

NSCool calls them only at rho < rhocore (nn bremsstrahlung only above
neutron drip).  Pair, photo and synchrotron emission are off in our runs
(I_Neutrino_1.dat).
"""
from __future__ import annotations

import functools
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from ._spline import spline_y2, splint
from .core_nu import _r_nn
from .superfluid import below, f32, fexp, u_1s0, u_3p2b

__all__ = ["eebrem", "npb", "plasma", "brem_nn_crust", "nalpha_table"]


@functools.lru_cache(maxsize=None)
def nalpha_table(path=None):
    """NSCool's Code/Data_Files/Nalpha.dat, read from the NSCool build
    (not shipped: NSCool has no licence).  x = log10 T9 + 9."""
    if path is None:
        from ..runner import default_home
        path = default_home() / "Code" / "Data_Files" / "Nalpha.dat"
    v = np.array(Path(path).read_text().split(), float).reshape(-1, 2)
    x = np.log10(v[:, 0]) + 9.0
    return x, v[:, 1], spline_y2(x, v[:, 1])


def eebrem(T, kfe, table=None):
    """Electron-electron pair bremsstrahlung (neebrem).

    NSCool evaluates the table at log10(T / (mu_e / 10 MeV)), with T in K.
    """
    x, y, y2 = nalpha_table() if table is None else table
    ok = kfe > 0
    mu = jnp.where(ok, kfe, 1.0) * 197.0
    naa = splint(x, y, y2, jnp.log10(T / (mu / 10.0)))
    return jnp.where(ok, 2.16e14 * (T / 1e9) ** 7 * (mu / 10.0) ** 2 * naa, 0.0)


_NPB = [f32(c) for c in (11.204, 7.304, 0.2976, 0.370, 0.188, 0.103, 0.0547, 6.77, 0.228)]


def npb(T, rho):
    """Electron-nucleus pair bremsstrahlung, Kaminker+1999 fit (npb_new)."""
    c0, c1, c2, c3, c4, c5, c6, c7, c8 = _NPB
    tau = jnp.log10(T / 1e8)
    r = jnp.log10(rho / 1e12)
    lgq = (c0 + c1 * tau + c2 * r - c3 * tau ** 2 + c4 * tau * r - c5 * r ** 2
           + c6 * tau ** 2 * r - c7 * jnp.log10(1.0 + c8 * rho / 2.8e14))
    return 10.0 ** lgq


def plasma(T, rho, A, Z):
    """Plasmon decay (nplasma); A is NSCool's a_cell, Z the nuclear charge."""
    ok = (Z != 0.0) & (A > 0)
    zs = jnp.where(ok, Z, 1.0)
    rza = rho * zs / jnp.where(ok, A, 1.0)
    l = T / 5.9302e9
    den = 1.0 + (1.019e-6 * rza) ** (2.0 / 3.0)
    g2 = 1.1095e11 * rza / T ** 2 / den ** 0.5
    g = jnp.sqrt(g2)
    f_t = 2.4 + 0.6 * g ** 0.5 + 0.51 * g + 1.25 * g ** 1.5
    f_l = (8.6 * g2 + 1.35 * g ** 3.5) / (225.0 - 17.0 * g + g2)
    x = (17.5 + jnp.log10(2.0 * rza) - 3.0 * jnp.log10(T)) / 6.0
    y = (-24.5 + jnp.log10(2.0 * rza) + 3.0 * jnp.log10(T)) / 6.0
    first = (0.39 - 1.25 * x - 0.35 * jnp.sin(4.5 * x)
             - 0.3 * fexp(-(4.5 * x + 0.9) ** 2))
    second = jnp.minimum(0.0, y - 1.6 + 1.25 * x) / (0.57 - 0.25 * x)
    f_xy = jnp.where((jnp.abs(x) > 0.7) | (y < 0.0), 1.0, 1.05 + first * fexp(-second ** 2))
    q = 3.00e21 * l ** 9 * g ** 6 * fexp(-g) * (f_l + f_t) * f_xy
    return jnp.where(ok, q, 0.0)


def brem_nn_crust(T, kfn, mstn, tcn, v_ion, i, isf):
    """nn bremsstrahlung of the dripped neutrons (nubrem_crust_nn), reduced by
    the volume fraction of nuclei v_ion."""
    q = 3.0 * 7.4e19 * mstn ** 4 * (kfn / 1.68) * 0.59 * 0.56 * (T / 1e9) ** 8
    m, tau = below(T, tcn)
    u = jnp.where(i < isf, u_3p2b(tau), u_1s0(tau))
    return jnp.where(m, _r_nn(u), 1.0) * q * (1.0 - v_ion)
