"""Unpaired inner-core thermal integrals on a native spherical TOV profile.

The caller chooses a density cut inside homogeneous matter. Native log-density
interpolation is used throughout; crust and pairing are outside this interface.
"""
from __future__ import annotations

from typing import NamedTuple
import jax
import jax.numpy as jnp
from . import core_nu, heat


class InnerCoreSums(NamedTuple):
    C: jax.Array              # erg/K, d E_infinity / d T_infinity
    L_nu: jax.Array           # erg/s at infinity
    L_durca: jax.Array        # electron + muon direct Urca at infinity
    volume_cm3: jax.Array     # proper volume of the selected inner core


def density_at_h(table, h):
    """Native EOS density (fm^-3), log-linear in pseudo-enthalpy."""
    return jnp.exp(jnp.interp(h, table.h, jnp.log(table.n)))


def _composition_at(comp, n):
    return tuple(jnp.interp(n, comp.n, field) for field in
                 (comp.x_p, comp.x_e, comp.x_mu, comp.mst_n, comp.mst_p))


def _kf(n, fraction):
    present = fraction > 0
    return jnp.where(present, (3.*jnp.pi**2*n*jnp.where(present, fraction, 1.))**(1./3.), 0.)


def _margin(comp, n, lepton):
    xp, xe, xm, _, _ = _composition_at(comp, n)
    return _kf(n, xp) + _kf(n, xe if lepton == 'electron' else xm) - _kf(n, 1.-xp)


def onset_density(comp, n_min, n_max, *, lepton='electron'):
    """First opening for one monotone Urca channel; n_max if still closed.

    Uses the interpolated composition actually supplied to the rate. The scalar
    root has an implicit derivative, including composition-node sensitivities.
    Re-closing/tangent roots are outside this first inner-core interface.
    """
    if lepton not in ('electron', 'muon'):
        raise ValueError("lepton must be electron or muon")
    grid = jnp.clip(comp.n, n_min, n_max)
    margin = _margin(comp, grid, lepton)
    opened = margin > 0
    k = jnp.maximum(jnp.argmax(opened), 1)
    a, b = grid[k-1], grid[k]
    f = lambda n: _margin(comp, n, lepton)

    def root(_):
        def solve(fun, initial):
            def step(i, bracket):
                lo, hi = bracket
                mid = .5*(lo+hi)
                positive = fun(mid) > 0
                return jnp.where(positive, lo, mid), jnp.where(positive, mid, hi)
            lo, hi = jax.lax.fori_loop(0, 48, step, (a, b))
            return .5*(lo+hi)
        return jax.lax.custom_root(f, .5*(a+b), solve, lambda g, y: y/g(jnp.ones_like(y)))

    return jax.lax.cond(f(n_min)>0, lambda _: n_min,
                        lambda _: jax.lax.cond(jnp.any(opened), root, lambda _: n_max, None), None)


def _shell_integral(r, integrand, volume_limit):
    """Partial-cell integral in r^3, so a moving cutoff retains its derivative."""
    v = r**3
    left, right = v[:-1], jnp.minimum(v[1:], volume_limit)
    width = jnp.maximum(right-left, 0.)
    fraction = width/jnp.diff(v)
    end = integrand[:-1] + fraction*jnp.diff(integrand)
    return 4.*jnp.pi/3.*jnp.sum(.5*(integrand[:-1]+end)*width)*1e15


def inner_core_sums(table, profile, h_c, comp, T_inf, *, n_cut):
    """C and L_nu of unpaired npeμ matter above n_cut, from a native profile.

    Requires a smooth EOS, supplied central h matching the profile, composition
    coverage, and at most one non-tangent opening per Urca channel inside the
    star. Saved-profile and shell-quadrature convergence must be checked.
    n_cut must lie above the model's crust remapping region; this is NOT a
    whole-star cooling curve. Density is the native-grid ansatz, not inferred
    from a renormalized surface chemical potential.
    """
    h = jnp.concatenate((jnp.atleast_1d(h_c), profile.h))
    r = jnp.concatenate((jnp.zeros(1), profile.r_km))
    m = jnp.concatenate((jnp.zeros(1), profile.m_km))
    ephi = profile.ephi[-1]*jnp.exp(profile.h[-1]-h)
    n = density_at_h(table, h)
    xp, xe, xm, mn, mp = _composition_at(comp, n)
    kn, kp, ke, km = (_kf(n, x) for x in (1.-xp, xp, xe, xm))
    t = T_inf/ephi
    metric = jnp.sqrt(1.-2.*m/jnp.where(r>0,r,1.))
    # Interpolate in sqrt(enthalpy drop); the central-series segment then
    # retains r ~ sqrt(h_c-h), including an Urca ball smaller than r_start.
    u = jnp.concatenate((jnp.zeros(1), jnp.sqrt(h_c-profile.h)))
    def volume_limit(density):
        target_h = jnp.interp(jnp.log(density), jnp.log(table.n), table.h)
        drop = h_c-target_h
        target_u = jnp.where(drop>0,jnp.sqrt(jnp.where(drop>0,drop,1.)),0.)
        return jnp.interp(target_u, u, r)**3
    limit = volume_limit(n_cut)
    integrate = lambda value, v=limit: _shell_integral(r, value/metric, v)
    cv = (heat.cv_coeff(kn,939.56*mn) + heat.cv_coeff(kp,938.27*mp)
          + heat.lepton_cv_coeff(ke,.511) + heat.lepton_cv_coeff(km,105.))*t
    qslow = (core_nu.murca(t,ke,km,kn,kp,mn,mp,0.,0.,0,-1)
             + core_nu.brem(t,kn,kp,mn,mp,0.,0.,0,-1))
    fast = jnp.asarray(0.)
    single_onset = jnp.asarray(True)
    for lepton, fraction in [('electron',xe), ('muon',xm)]:
        opened = _margin(comp,jnp.clip(comp.n,n_cut,n[0]),lepton)>0
        single_onset &= ~jnp.any(opened[:-1] & ~opened[1:])
        threshold = onset_density(comp,n_cut,n[0],lepton=lepton)
        active = _margin(comp,n[0],lepton)>0
        v = jax.lax.cond(active, lambda _: volume_limit(threshold), lambda _: jnp.asarray(0.), None)
        q = core_nu.durca_unpaired_channel(t,mn,mp,n,fraction)
        fast += integrate(ephi**2*q,v)
    valid = (n[0]>n_cut) & (n_cut>=comp.n[0]) & (n[0]<=comp.n[-1]) & single_onset & (T_inf>0)
    values = InnerCoreSums(integrate(cv),integrate(ephi**2*qslow)+fast,fast,integrate(jnp.ones_like(n)))
    return InnerCoreSums(*(jnp.where(valid,value,jnp.nan) for value in values))
