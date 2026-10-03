"""NSCool's physics summed over a star with uniform redshifted temperature.

``Star`` holds the zone arrays of an NSCool star (``Star.from_dump``; C3
builds them from nmbackend TOV and composition).  ``zone_terms`` gives
every emissivity and heat-capacity term per temperature zone, as
NSCool's ``neutrino`` and ``specheat``; ``isothermal_sums`` gives C,
L_nu, L_gamma and T_in as ``NSCool_iso.out`` does (C1 driver).
"""
from __future__ import annotations

import dataclasses
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import conduct, core_nu, crust_nu, envelope, heat
from .gaps import StarTc, star_tc

__all__ = ["Star", "zone_terms", "IsoSums", "isothermal_sums"]

_LSOL = 3.826e33
_SIGMA = 5.67e-5
_PI = 3.1415926535

_ARRAYS = ("rad", "rho", "nb", "debar", "dvol", "ephi", "a2ephin", "Ye", "Ymu",
           "kfe", "kfm", "kfn", "kfp", "mstn", "mstp", "A_cell", "A_ion", "Z_ion",
           "v_ion")


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Star:
    """Zone arrays i = 0..imax+1 (centre first), NSCool units (cm, g/cm^3,
    fm^-3, fm^-1); temperatures live on odd zones."""
    rad: jnp.ndarray
    rho: jnp.ndarray
    nb: jnp.ndarray
    debar: jnp.ndarray
    dvol: jnp.ndarray
    ephi: jnp.ndarray
    a2ephin: jnp.ndarray
    Ye: jnp.ndarray
    Ymu: jnp.ndarray
    kfe: jnp.ndarray
    kfm: jnp.ndarray
    kfn: jnp.ndarray
    kfp: jnp.ndarray
    mstn: jnp.ndarray
    mstp: jnp.ndarray
    A_cell: jnp.ndarray
    A_ion: jnp.ndarray
    Z_ion: jnp.ndarray
    v_ion: jnp.ndarray
    gs14: jnp.ndarray
    imax: int = dataclasses.field(metadata=dict(static=True))
    icore: int = dataclasses.field(metadata=dict(static=True))
    idrip: int = dataclasses.field(metadata=dict(static=True))
    rhocore: float = dataclasses.field(metadata=dict(static=True))
    rhodrip: float = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def from_dump(cls, d) -> "Star":
        """From ``NSCool.zone_dump``."""
        a = {k: jnp.asarray(d.star[k], float) for k in _ARRAYS}
        return cls(**a, gs14=jnp.asarray(d.gs14), imax=d.imax, icore=d.icore,
                   idrip=d.idrip, rhocore=d.rhocore, rhodrip=d.rhodrip)

    @property
    def zones(self) -> np.ndarray:
        return np.arange(1, self.imax + 1, 2)

    def tc(self, gaps, **kw) -> StarTc:
        """Gap Tc per zone (``gaps.star_tc``)."""
        return star_tc(gaps, self.kfn, self.kfp, self.nb, self.icore, self.idrip, **kw)


def zone_terms(T, s: Star, tc: StarTc, ca_n3p2=0.5, qimp=1e-10) -> dict:
    """Per-zone terms at redshifted T on the temperature zones (``Star.zones``).

    Keys follow ``NSCool.zone_dump``: q* (erg cm^-3 s^-1), cv* (erg cm^-3 K^-1),
    lam (crust zones only; 0 in the core, which is C4).
    """
    i = jnp.asarray(s.zones)
    g = lambda a: a[i]
    t = T / g(s.ephi)
    rho = g(s.rho)
    core = i <= s.icore                  # rho >= rhocore
    crust = ~core
    inner = crust & (rho >= s.rhodrip)
    tcn, tcp, isf = g(tc.tcn), g(tc.tcp), tc.isf
    kfe, kfm, kfn, kfp = g(s.kfe), g(s.kfm), g(s.kfn), g(s.kfp)
    mstn, mstp = g(s.mstn), g(s.mstp)
    A = jnp.where(crust, g(s.A_cell), 1.0)
    Z = jnp.where(crust, g(s.Z_ion), 1.0)
    kfe_c = jnp.where(crust, kfe, 1.0)
    z0 = jnp.zeros_like(t)
    q = {
        "qeebrem": jnp.where(crust, crust_nu.eebrem(t, kfe_c), 0.0),
        "qnpb": jnp.where(crust, crust_nu.npb(t, rho), 0.0),
        "qplasma": jnp.where(crust, crust_nu.plasma(t, rho, A, Z), 0.0),
        "qbrem_nn": jnp.where(inner, crust_nu.brem_nn_crust(
            t, kfn, mstn, tcn, g(s.v_ion), i, isf), 0.0),
        "qmurca_nucl": jnp.where(core, core_nu.murca(
            t, kfe, kfm, kfn, kfp, mstn, mstp, tcn, tcp, i, isf), 0.0),
        "qbrem_nucl": jnp.where(core, core_nu.brem(
            t, kfn, kfp, mstn, mstp, tcn, tcp, i, isf), 0.0),
        "qdurca_np": jnp.where(core, core_nu.durca(
            t, kfe, kfm, kfn, kfp, mstn, mstp, g(s.nb), g(s.Ye), g(s.Ymu),
            tcn, tcp, i, isf), 0.0),
        "qpbf_n1s0": jnp.where(i > isf, core_nu.pbf_1s0(t, tcn, mstn, kfn), 0.0),
        "qpbf_n3p2": jnp.where(i <= isf, core_nu.pbf_n3p2(t, tcn, mstn, kfn, ca_n3p2), 0.0),
        "qpbf_p1s0": core_nu.pbf_1s0(t, tcp, jnp.where(core, mstp, 1.0), kfp),
    }
    q["qnu"] = sum(q.values()) + z0
    cn = heat.cv_coeff(kfn, 939.56 * mstn)
    cp = heat.cv_coeff(kfp, 938.27 * mstp)
    c = {
        "cv_n": jnp.where(rho >= s.rhodrip, heat.cv_neutron(t, cn, tcn, i, isf), 0.0),
        "cv_p": jnp.where(core, heat.cv_proton(t, cp, tcp), 0.0),
        "cv_e": heat.lepton_cv_coeff(kfe, 0.511) * t,
        "cv_m": heat.lepton_cv_coeff(kfm, 105.0) * t,
        "cv_ion": jnp.where(crust, heat.cv_ion(t, rho, A, Z), 0.0),
    }
    c["cv"] = sum(c.values())
    rho_c = jnp.where(crust, rho, s.rhodrip)
    lam = jnp.where(crust, conduct.lambda_crust(t, rho_c, A, Z, kfe_c, qimp, s.rhodrip), 0.0)
    return {**q, **c, "T": t, "lam": lam}


class IsoSums(NamedTuple):
    C: jnp.ndarray          # erg/K
    L_nu: jnp.ndarray       # erg/s at infinity
    L_gamma: jnp.ndarray    # erg/s at infinity
    Teff_inf: jnp.ndarray
    T_in: jnp.ndarray       # redshifted T at the march's inner end
    C_core: jnp.ndarray
    L_durca: jnp.ndarray
    L_nu_crust: jnp.ndarray


def _l_gamma(T, s: Star, eta):
    """Photon luminosity at infinity and Teff_inf, as the C1 driver."""
    im = s.imax
    radius = s.rad[im]
    teff = envelope.teff_local(T / s.ephi[im], eta, s.gs14)
    lg = 4.0 * _PI * radius ** 2 * _SIGMA * teff ** 4 * s.ephi[im] ** 2
    te_inf = ((lg / s.ephi[im - 1] ** 2 / (4.0 * _PI * radius ** 2 * _SIGMA)) ** 0.25
              * s.ephi[im - 1])
    return lg, te_inf


def _fp(s: Star, i, ttil, qimp):
    """Flux-equation coefficient lam bar / L_sun at zone i.

    Crust formula everywhere.  The march reaches one core zone, i = icore
    at rho_b = rhocore; there it uses the composition of zone icore + 2,
    while NSCool uses its core conductivity (C4), and T_in differs by up
    to 0.15%.  Deeper zones (steps the march skips) are clamped to icore
    so that their unused values and gradients stay finite.
    """
    i = jnp.maximum(i, s.icore)
    j = jnp.maximum(i, s.icore + 2)
    t = ttil / s.ephi[i]
    lam = conduct.lambda_crust(t, s.rho[i], s.A_cell[j], s.Z_ion[j], s.kfe[i], qimp, s.rhodrip)
    return lam * s.nb[i] / _LSOL


def crust_march(T, s: Star, lg, rho_b, qimp=1e-10, n_iter=20):
    """T_in: march the steady flux equation from T_b = T at zone imax inward
    to the first zone with rho >= rho_b, at L = lg (C1 driver).

    NSCool's last step reaches zone icore when rho_b = rhocore; there the
    core conductivity is C4 work, so the crust formula is used at that zone.
    """
    xl = lg / _LSOL

    def step(carry, j1):
        tcur, fpout = carry
        j0, ie = j1 - 2, j1 - 1
        num = xl * (s.debar[ie] + s.debar[ie + 1])

        def picard(c, _):
            x0, done = c
            x1 = tcur + num / (0.5 * (_fp(s, j0, x0, qimp) + fpout) * s.a2ephin[ie])
            new_done = done | (jnp.abs(x1 - x0) < 1e-10 * x1)
            return (jnp.where(done, x0, x1), new_done), None

        (x, _), _ = jax.lax.scan(picard, (tcur, jnp.asarray(False)), None, length=n_iter)
        on = s.rho[j1] < rho_b          # rho grows inwards, so steps stop once
        return (jnp.where(on, x, tcur), jnp.where(on, _fp(s, j0, x, qimp), fpout)), None

    (tin, _), _ = jax.lax.scan(step, (T, _fp(s, s.imax, T, qimp)),
                               jnp.arange(s.imax, 1, -2))
    return tin


def isothermal_sums(T, s: Star, tc: StarTc, eta=0.0, rho_b=0.0, ca_n3p2=0.5,
                    qimp=1e-10) -> IsoSums:
    """C, L_nu, L_gamma at uniform redshifted T (scalar), as NSCool_iso.out.

    ``eta`` is PCY97's eta (NSCool's ETA / 0.443).  ``rho_b`` > rhosurf
    marches the crust (``crust_march``); otherwise T_in = T.
    """
    z = jnp.asarray(s.zones)
    terms = zone_terms(T, s, tc, ca_n3p2, qimp)
    dv = s.dvol[z] + s.dvol[z + 1]
    w = s.ephi[z] ** 2 * dv
    core = z <= s.icore
    lg, te = _l_gamma(T, s, eta)
    tin = crust_march(T, s, lg, rho_b, qimp)
    return IsoSums(jnp.sum(terms["cv"] * dv), jnp.sum(terms["qnu"] * w), lg, te, tin,
                   jnp.sum(jnp.where(core, terms["cv"] * dv, 0.0)),
                   jnp.sum(terms["qdurca_np"] * w),
                   jnp.sum(jnp.where(core, 0.0, terms["qnu"] * w)))
