"""Moving density contacts and proper-volume quadrature on native TOV profiles."""
from __future__ import annotations
from typing import NamedTuple

import jax
import jax.numpy as jnp
from numpy.polynomial.legendre import leggauss


def enthalpy_at_energy(table, energy):
    """Invert actual Hermite eps(h) for a scalar energy in km^-2; unsupported roots give NaN.

    Requires strictly ordered finite nodes and a non-tangent monotone selected
    interval. Derivatives include EOS nodes and target energy, conditional on
    this interval. At an EOS knot the lower interval supplies the derivative.
    """
    if table.h.size < 2:
        raise ValueError('EOS needs at least two nodes')
    i = jnp.clip(jnp.searchsorted(table.eps, energy, side='left'), 1, table.h.size-1)
    lo, hi = table.h[i-1], table.h[i]
    dh = hi-lo
    safe_dh = jnp.where(dh > 0, dh, 1.)
    p0, p1 = table.p[i-1], table.p[i]
    s0, s1 = table.eps[i-1]+p0, table.eps[i]+p1
    c1 = safe_dh*s0
    c2 = 3*(p1-p0)-safe_dh*(2*s0+s1)
    c3 = 2*(p0-p1)+safe_dh*(s0+s1)
    # eps = p_t/dh-p is cubic in t; its h derivative is quadratic.
    e1, e2, e3 = 2*c2/safe_dh-c1, 3*c3/safe_dh-c2, -c3
    vertex = jnp.clip(-e2/jnp.where(e3 != 0, 3*e3, 1.), 0., 1.)
    derivative = lambda t: (e1+2*e2*t+3*e3*t*t)/safe_dh
    minimum = jnp.minimum(derivative(0.), jnp.minimum(derivative(1.), derivative(vertex)))
    scale = jnp.maximum(jnp.maximum(jnp.abs(table.eps[i-1]), jnp.abs(table.eps[i])), 1e-300)
    tolerance = 64*jnp.finfo(table.h.dtype).eps*scale
    f = lambda h: table.at_h(h)[0]-energy
    nodes_valid = (jnp.all(jnp.isfinite(table.h)) & jnp.all(jnp.isfinite(table.eps))
                   & jnp.all(jnp.isfinite(table.p)) & jnp.all(jnp.diff(table.h) > 0)
                   & jnp.all(jnp.diff(table.eps) > 0) & jnp.all(table.eps > 0)
                   & jnp.all(table.p > 0))
    valid = (nodes_valid & jnp.isfinite(energy) & (energy > 0)
             & (energy >= table.eps[0]) & (energy <= table.eps[-1])
             & (f(lo) <= tolerance) & (f(hi) >= -tolerance) & jnp.isfinite(minimum)
             & (minimum > 1e-12*scale/safe_dh))

    def root(_):
        def solve(fun, initial):
            def step(_, bracket):
                a, b = bracket
                mid = .5*(a+b)
                above = fun(mid) > 0
                return jnp.where(above, a, mid), jnp.where(above, mid, b)
            a, b = jax.lax.fori_loop(0, 48, step, (lo, hi))
            return .5*(a+b)
        h = jax.lax.custom_root(f, .5*(lo+hi), solve, lambda g, y: y/g(jnp.ones_like(y)))
        supported = jnp.isfinite(h) & (jnp.abs(f(h)) <= tolerance)
        return jnp.where(supported, h, jnp.nan)

    return jax.lax.cond(valid, root, lambda _: jnp.asarray(jnp.nan), None)


def _profile_nodes(profile, h_c):
    drop = h_c-profile.h
    u = jnp.concatenate((jnp.zeros(1), jnp.sqrt(jnp.where(drop > 0, drop, 1.))))
    r = jnp.concatenate((jnp.zeros(1), profile.r_km))
    m = jnp.concatenate((jnp.zeros(1), profile.m_km))
    valid = (jnp.isfinite(h_c) & jnp.all(jnp.isfinite(profile.h))
             & jnp.all(jnp.isfinite(profile.r_km)) & jnp.all(jnp.isfinite(profile.m_km))
             & jnp.all(jnp.isfinite(profile.ephi)) & jnp.all(jnp.isfinite(drop)) & jnp.all(drop > 0)
             & jnp.all(jnp.diff(profile.h) < 0) & jnp.all(jnp.diff(r) > 0)
             & jnp.all(jnp.diff(m) >= 0) & jnp.all(profile.m_km >= 0)
             & jnp.all(profile.ephi > 0) & jnp.all(2*profile.m_km < profile.r_km))
    return u, r, m, valid


def volume_at_h(profile, h_c, h):
    """r^3 in km^3 at a supported h; h_c must match the profile-generating solve."""
    u, r, _, valid = _profile_nodes(profile, h_c)
    drop = h_c-h
    target = jnp.where(drop > 0, jnp.sqrt(jnp.where(drop > 0, drop, 1.)), 0.)
    volume = jnp.interp(target, u, r)**3
    valid &= jnp.isfinite(h) & (h >= profile.h[-1]) & (h <= h_c)
    return jnp.where(valid, volume, jnp.nan)


class VolumeGeometry(NamedTuple):
    h: jax.Array
    r_km: jax.Array
    m_km: jax.Array
    ephi: jax.Array
    proper_factor: jax.Array  # dV_proper / d(r^3), cm^3 / km^3


def geometry_at_volume(profile, h_c, volume):
    """Geometry at r^3; centre uses m~r^3 and lapse uses the exact first integral."""
    u, r, m, valid = _profile_nodes(profile, h_c)
    radius = jnp.where(volume > 0, jnp.where(volume > 0, volume, 1.)**(1./3.), 0.)
    h = h_c-jnp.interp(radius, r, u)**2
    mass = jnp.interp(volume, r**3, m)
    ephi = profile.ephi[-1]*jnp.exp(profile.h[-1]-h)
    compactness = 2*mass/jnp.where(radius > 0, radius, 1.)
    factor = 4*jnp.pi/3*1e15/jnp.sqrt(jnp.where(compactness < 1, 1-compactness, 1.))
    valid &= (jnp.isfinite(volume) & (volume >= 0) & (volume <= r[-1]**3)
              & jnp.isfinite(factor) & jnp.isfinite(h) & jnp.isfinite(radius)
              & jnp.isfinite(mass) & jnp.isfinite(ephi) & (compactness < 1))
    return VolumeGeometry(*(jnp.where(valid, x, jnp.nan) for x in (h, radius, mass, ephi, factor)))


def volume_quadrature(profile, h_c, edges, *, order=8):
    """Gauss geometry and proper-volume weights for ascending contact r^3 edges.

    Equal adjacent edges give finite nodes and zero weights. Endpoint motion
    remains differentiable; contact ordering and stellar support must be fixed.
    h_c must match the profile-generating solve. Invalid edges give NaN weights.
    No thermal redshift weights are included: callers supply their rate's factor.
    Refine order and saved-profile resolution together for each thermal problem.
    """
    if edges.size < 2:
        raise ValueError('quadrature needs at least two edges')
    if order < 1:
        raise ValueError('order must be positive')
    nodes, weights = (jnp.asarray(x) for x in leggauss(order))
    width = jnp.diff(edges)
    volume = edges[:-1, None]+.5*width[:, None]*(nodes+1)
    geometry = geometry_at_volume(profile, h_c, volume)
    valid = (jnp.all(jnp.isfinite(edges)) & jnp.all(width >= 0)
             & jnp.all(edges >= 0) & jnp.all(edges <= profile.r_km[-1]**3))
    proper = .5*width[:, None]*weights*geometry.proper_factor
    return geometry, jnp.where(valid, proper, jnp.nan)
