"""Envelope T_b -> T_s: Potekhin, Chabrier & Yakovlev 1997 (PCY97), Eqs. A7-A10.

As NSCool's ``fteff_acc``.  ``eta`` here is PCY97's eta = g_s14^2 Delta M / M.
NSCool divides its input ETA by 0.443 before this formula
(see physics/README.md, envelope normalization), so NSCool ETA = 0.443 eta.
Steiner interprets the fixed factor as a reference-gravity normalization;
it is not recomputed from the current star's gs14.
"""
from __future__ import annotations

import jax.numpy as jnp

from .superfluid import f32

__all__ = ["teff_local", "gs14"]

_G, _MSUN, _C = 6.67e-8, 1.989e33, 2.99792e10


def gs14(mass_msun, radius_cm):
    """Surface gravity / 1e14 cm s^-2, as NSCool.f (with its constants)."""
    root = jnp.sqrt(1.0 - 2.0 * _G * _MSUN * mass_msun / radius_cm / _C ** 2)
    return _G * _MSUN * mass_msun / radius_cm ** 2 / root / 1e14


def teff_local(Tb, eta, g14):
    """Local effective temperature (K) for local T_b (K) at rho_b = 1e10 g/cm^3."""
    tb9 = Tb / 1e9
    ts = jnp.sqrt(7.0 * tb9 * jnp.sqrt(g14))
    z = tb9 - ts / 1e3
    t4_iron = g14 * ((7.0 * z) ** 2.25 + (z / 3.0) ** 1.25)
    t4_wacc = g14 * (18.1 * tb9) ** f32(2.42)
    acc = eta > 1e-30
    es = jnp.where(acc, eta, 1.0)
    a = (1.2 + (5.3e-6 / es) ** f32(0.38)) * tb9 ** f32(5.0 / 3.0)
    t4 = jnp.where(acc, (a * t4_iron + t4_wacc) / (a + 1.0), t4_iron)
    return t4 ** 0.25 * 1e6
