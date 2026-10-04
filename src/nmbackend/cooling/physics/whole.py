"""Internal forward-only unpaired core/crust sums on a native spherical profile.

The operational density cut and tabulated crust prescription are explicit.
Integrated derivatives remain unqualified until ionic moving contacts are
included; this module is not a cooling curve or transport solver.
"""
from __future__ import annotations

from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np

from ...constants import MEVFM3_TO_GEOM
from ...eos.crust import _G_CM3_TO_MEVFM3
from . import heat
from .crust import crust_chemistry, _unpaired_crust_terms
from .geometry import enthalpy_at_energy, volume_at_h, volume_quadrature
from .native import (InnerCoreSums, density_at_h, onset_density, _margin,
                     _unpaired_core_terms, _composition_at)
from .superfluid import f32


class CrustSums(NamedTuple):
    """Crust budget with ionic capacity separated for phase-fit qualification."""
    C: jax.Array
    L_nu: jax.Array
    C_ion: jax.Array
    volume_cm3: jax.Array


class WholeStarSums(NamedTuple):
    """Separate core/crust budgets; capacities in erg/K, luminosities in erg/s."""
    core: InnerCoreSums
    crust: CrustSums

    @property
    def C(self):
        return self.core.C+self.crust.C

    @property
    def L_nu(self):
        return self.core.L_nu+self.crust.L_nu

    @property
    def L_durca(self):
        return self.core.L_durca

    @property
    def volume_cm3(self):
        return self.core.volume_cm3+self.crust.volume_cm3


def _unpaired_sums(table, profile, h_c, comp, crust, T_inf, *,
                   rho_core=1.6e14, rho_base=1e10, rho_drip=4e11, order=16):
    """Instantaneous forward sums down to rho_base, excluding the outer blanket.

    Requires matched profile/h_c, composition coverage, a supported chemistry
    join and at most one simple opening per Urca channel. Density boundaries
    are static prescription choices. EOS/profile, chemistry and Urca contacts
    split quadrature; ionic phase jumps require forward order refinement and
    their moving-boundary derivatives are NOT qualified by this interface.
    Callers must qualify absence of the ionic source-table overrun region
    between sampled nodes; the local guard alone is not a coverage proof.
    Homogeneous core composition is continued down to the operational core
    cut, including any EOS joining segment. This is a declared ansatz.
    No gradients are stopped. Invalid stellar/composition support gives NaN.
    """
    if not 0 < rho_base < rho_drip < rho_core:
        raise ValueError('Require 0 < rho_base < rho_drip < rho_core')
    conversion = _G_CM3_TO_MEVFM3*MEVFM3_TO_GEOM
    # Static prescription nodes; EOS-dependent contact positions are evaluated
    # below in JAX. Keep repeated clipped edges as finite zero-weight padding.
    densities = jnp.sort(jnp.concatenate((jnp.asarray([
        rho_base, rho_drip, np.clip(f32(4.3e11), rho_base, rho_core), rho_core]),
        jnp.clip(crust.rho, rho_base, rho_core))))
    hs = jax.vmap(lambda rho: enthalpy_at_energy(table, rho*conversion))(densities)
    hbase, hcore = hs[0], hs[-1]
    ncore, nc = density_at_h(table, hcore), density_at_h(table, h_c)
    yecore = jnp.interp(ncore, comp.n, comp.x_e)
    onsets, topology = [], jnp.asarray(True)
    for channel in ('electron', 'muon'):
        opened = _margin(comp, jnp.clip(comp.n, ncore, nc), channel) > 0
        topology &= ~jnp.any(opened[:-1] & ~opened[1:])
        onsets.append(onset_density(comp, ncore, nc, lepton=channel))
    hdu = jnp.interp(jnp.log(jnp.asarray(onsets)), jnp.log(table.n), table.h)
    hcomp = jnp.interp(jnp.log(jnp.clip(comp.n, ncore, nc)), jnp.log(table.n), table.h)
    core_h = jnp.sort(jnp.clip(jnp.concatenate((jnp.asarray([h_c, hcore]),
        table.h, profile.h, hcomp, hdu)), hcore, h_c))[::-1]
    crust_h = jnp.sort(jnp.clip(jnp.concatenate((hs, table.h, profile.h)), hbase, hcore))[::-1]
    core_edges = volume_at_h(profile, h_c, core_h)
    crust_edges = volume_at_h(profile, h_c, crust_h)
    g, w = volume_quadrature(profile, h_c, core_edges, order=order)
    n = density_at_h(table, g.h)
    cv, slow, qe, qm = _unpaired_core_terms(T_inf/g.ephi, n, comp)
    resolved = jnp.asarray(True)
    for channel in ('electron', 'muon'):
        opened = _margin(comp, n, channel) > 0
        resolved &= ~jnp.any((w[:, 0] > 0) & jnp.any(opened[:, 1:] != opened[:, :-1], axis=1))
    fast = jnp.where(_margin(comp, n, 'electron') > 0, qe, 0.)
    fast += jnp.where(_margin(comp, n, 'muon') > 0, qm, 0.)
    xp, xe, xm, mn, mp = _composition_at(comp, n)
    physical = (jnp.all(jnp.isfinite(jnp.stack((xp, xe, xm, mn, mp))))
                & jnp.all((xp >= 0) & (xp <= 1) & (xe >= 0) & (xm >= 0))
                & jnp.all(jnp.abs(xp-xe-xm) <= 1e-12)
                & jnp.all((mn > 0) & (mp > 0)))
    core = InnerCoreSums(jnp.sum(w*cv), jnp.sum(w*g.ephi**2*(slow+fast)),
                         jnp.sum(w*g.ephi**2*fast), jnp.sum(w))
    gc, wc = volume_quadrature(profile, h_c, crust_edges, order=order)
    rho = table.at_h(gc.h)[0]/conversion
    # Padded cells can sit exactly on a rounded chemistry endpoint. Evaluate
    # their unused rates in a covered interior state; their weights remain zero.
    rho = jnp.where(wc > 0, rho, np.sqrt(rho_base*rho_core))
    chemistry = crust_chemistry(crust, rho, core_rho=rho_core,
                                core_n=ncore, core_Ye=yecore)
    cvc, qc = _unpaired_crust_terms(T_inf/gc.ephi, rho, chemistry, rho_drip=rho_drip)
    gamma = heat._G*chemistry.Z**2*(rho/chemistry.A_cell)**heat._THIRD/(T_inf/gc.ephi)
    a1 = jnp.where(rho >= f32(4.3e11), 3*chemistry.Z, chemistry.A_cell)
    delta = heat._NA*chemistry.Z*jnp.sqrt(rho/(a1*chemistry.A_cell))/(T_inf/gc.ephi)
    source_supported = ~jnp.any((wc > 0) & (gamma > 178) & (gamma <= 210)
                                & (delta > f32(7e20)))
    ion = heat.cv_ion(T_inf/gc.ephi, rho, chemistry.A_cell, chemistry.Z)
    shell = CrustSums(jnp.sum(wc*cvc), jnp.sum(wc*gc.ephi**2*qc),
                      jnp.sum(wc*ion), jnp.sum(wc))
    valid = (jnp.isfinite(T_inf) & (T_inf > 0) & topology & resolved & physical & source_supported
             & jnp.all(jnp.isfinite(comp.n)) & jnp.all(comp.n > 0)
             & jnp.all(jnp.diff(comp.n) > 0)
             & jnp.all(jnp.stack([jnp.all(jnp.isfinite(x) & (x >= 0))
                                 for x in (w, cv, slow, qe, qm, wc, cvc, qc, ion)]))
             & (hbase < hcore) & (hcore < h_c)
             & (ncore >= comp.n[0]) & (nc <= comp.n[-1])
             & jnp.all(jnp.isfinite(hs))
             & jnp.all(jnp.isfinite(jnp.asarray(core)))
             & jnp.all(jnp.isfinite(jnp.asarray(shell))))
    return WholeStarSums(*(type(sums)(*(jnp.where(valid, x, jnp.nan) for x in sums))
                           for sums in (core, shell)))
