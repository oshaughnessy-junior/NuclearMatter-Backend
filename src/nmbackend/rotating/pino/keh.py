"""JAX KEH Green map and hydrostationary closure, in physical K=G=c=1 units.

Independently implemented from Ray, arXiv:2611.05790v1, Eqs. 15–36.
Quadrature uses compactified-radius Gauss nodes; the exterior gravitational
sources are retained. Equilibrium fields and RNS references are separate.
"""
from __future__ import annotations

from typing import NamedTuple
import numpy as np
from numpy.polynomial import legendre
from scipy.special import eval_gegenbauer
import jax
import jax.numpy as jnp


class Grid(NamedTuple):
    s: jax.Array
    mu: jax.Array
    ws: jax.Array
    wm: jax.Array
    green: jax.Array
    angular: jax.Array
    alpha_integral: jax.Array


class Fields(NamedTuple):
    value: jax.Array
    ds: jax.Array
    dm: jax.Array
    dss: jax.Array
    dsm: jax.Array
    dmm: jax.Array
    axis: jax.Array
    center: jax.Array
    pole: jax.Array
    equator: jax.Array
    re: jax.Array
    omega: jax.Array
    hc: jax.Array
    validity_penalty: jax.Array


def make_grid(nr=32, nmu=12, harmonics=4):
    """Precompute three Gegenbauer Green inverses including Jacobians once."""
    if nr < 8 or nmu < 4 or harmonics < 1 or 2*harmonics > nmu:
        raise ValueError('Need nr>=8, nmu>=4, 1<=harmonics<=nmu/2')
    s, ws = legendre.leggauss(nr)
    mu, wm = legendre.leggauss(nmu)
    s, ws, mu, wm = (s+1)/2, ws/2, (mu+1)/2, wm/2
    x = s/(1-s)
    low = np.minimum(x[:, None], x[None, :])
    high = np.maximum(x[:, None], x[None, :])
    greens, angular = [], []
    for d in (4, 3, 5):  # gamma, rho, omega
        pieces = []
        basis = []
        for mode in range(harmonics):
            ell = 2*mode
            y = eval_gegenbauer(ell, (d-2)/2, mu)
            norm = (1/(4*mode+1) if d == 3 else
                    np.pi/4 if d == 4 else (2*mode+1)*(2*mode+2)/(4*mode+3))
            radial = -(low/high)**ell/high**(d-2)
            radial *= (x**(d-1)/(1-s)**2*ws)[None, :]
            projection = wm*(1-mu**2)**((d-3)/2)*y/((2*ell+d-2)*norm)
            pieces.append(radial[:, :, None]*projection[None, None, :])
            basis.append(y)
        greens.append(np.stack(pieces))
        angular.append(np.stack(basis))
    # Integrate the polynomial through Gauss mu nodes from each node to the axis.
    vandermonde = legendre.legvander(2*mu-1, nmu-1)
    primitives = []
    for degree in range(nmu):
        coefficients = np.zeros(nmu)
        coefficients[degree] = 1
        integ = legendre.legint(coefficients)
        primitives.append((legendre.legval(1., integ)
                           - legendre.legval(2*mu-1, integ))/2)
    alpha_integral = np.stack(primitives, axis=1) @ np.linalg.inv(vandermonde)
    return Grid(*(jnp.asarray(a, dtype=jnp.float64) for a in
                  (s, mu, ws, wm, np.stack(greens), np.stack(angular), alpha_integral)))


def fields_from_evaluator(evaluate_scaled, hc, q, grid):
    """Decode scaled metric predictions and impose centre/pole/equator closure.

    Radius is solved jointly through the centre-pole relation for the scaled
    potentials. Illegal iterates receive a loss penalty; callers must reject
    them at validation, even if a clipped training evaluation remains finite.
    """
    center = evaluate_scaled(jnp.array([0., 0.]))
    pole = evaluate_scaled(jnp.array([q/(1+q), 1.]))
    equator = evaluate_scaled(jnp.array([.5, 0.]))
    delta = jnp.sum(pole[:2]-center[:2])
    re2 = 2*hc/jnp.maximum(delta, 1e-6)
    re = jnp.sqrt(re2)
    factors = jnp.array([re2, re2, 1/re])
    c, p, e = center*factors, pole*factors, equator*factors
    rotation_squared = -jnp.expm1(jnp.sum(p[:2]-e[:2]))
    omega = e[2] + jnp.exp(e[1])/re*jnp.sqrt(jnp.maximum(rotation_squared, 1e-16))
    # The exact static limit avoids differentiating sqrt(0) during training.
    omega = jnp.where(q == 1, 0., omega)
    coords = jnp.stack(jnp.meshgrid(grid.s, grid.mu, indexing='ij'), axis=-1)
    def sample(point):
        value = evaluate_scaled(point)
        first = jax.jacfwd(evaluate_scaled)(point)
        second = jax.jacfwd(jax.jacfwd(evaluate_scaled))(point)
        return value*factors, first[:, 0]*factors, first[:, 1]*factors, second[:, 0, 0]*factors, second[:, 0, 1]*factors, second[:, 1, 1]*factors
    arrays = jax.vmap(jax.vmap(sample))(coords)
    axis = jax.vmap(lambda s: evaluate_scaled(jnp.array([s, 1.]))*factors)(grid.s)
    penalty = jax.nn.relu(1e-5-delta)**2 + jax.nn.relu(-rotation_squared)**2
    # At s=0 all latitudes represent the same physical point.
    centers = jax.vmap(lambda mu: evaluate_scaled(jnp.array([0., mu]))*factors)(grid.mu)
    penalty += jnp.mean((centers-c)**2)
    return Fields(*arrays, axis, c, p, e, re, omega, hc, penalty)


def alpha_from_fields(fields, grid):
    """Axis-regular first-order alpha equation, including gamma second derivatives."""
    g, rho, w = jnp.moveaxis(fields.value, -1, 0)
    sm = grid.s[:, None]*(1-grid.s[:, None])
    mu = grid.mu[None, :]
    m = 1-mu**2
    r = fields.re*grid.s[:, None]/(1-grid.s[:, None])
    dg, drho, dw = jnp.moveaxis(sm[..., None]*fields.ds, -1, 0)
    gm, rhom, wm = jnp.moveaxis(fields.dm, -1, 0)
    d2g = sm*((1-2*grid.s[:, None])*fields.ds[..., 0] + sm*fields.dss[..., 0])
    dnu, num = (dg+drho)/2, (gm+rhom)/2
    a, b = 1+dg, -mu+m*gm
    denominator = m*a*a+b*b
    f = (.5*(d2g+dg*dg-m*fields.dmm[..., 0]+2*mu*gm-b*gm)
         + dnu*dnu-m*num*num
         - .25*m*r*r*jnp.exp(-2*rho)*(dw*dw-m*wm*wm))
    gterm = (-m*(2*dnu*num+sm*fields.dsm[..., 0]+dg*gm)
             + mu*dg + .5*m*m*r*r*jnp.exp(-2*rho)*dw*wm)
    source = -num-(b*f+a*gterm)/jnp.maximum(denominator, 1e-12)
    axis_alpha = (fields.axis[:, 0]-fields.axis[:, 1])/2
    alpha = axis_alpha[:, None] - source @ grid.alpha_integral.T
    return alpha


def matter_from_fields(fields, grid, eos):
    """Connected central fluid only; the exterior Bernoulli continuation is vacuum."""
    g, rho, omega = jnp.moveaxis(fields.value, -1, 0)
    r = fields.re*grid.s[:, None]/(1-grid.s[:, None])
    v = (fields.omega-omega)*r*jnp.sqrt(1-grid.mu[None, :]**2)*jnp.exp(-rho)
    candidate = grid.s[:, None] <= .5  # uniformly rotating oblate star lies inside r_e
    v2_for_h = jnp.where(candidate, jnp.minimum(v*v, 1-1e-8), 0.)
    h = jnp.sum(fields.pole[:2])/2 - (g+rho)/2 - .5*jnp.log1p(-v2_for_h)
    interior = candidate & (h > 0)
    interior = jnp.cumprod(interior.astype(jnp.int32), axis=0).astype(bool)
    eps, pressure, rest = eos(jnp.where(interior, h, 0.))
    eps, pressure, rest = (jnp.where(interior, a, 0.) for a in (eps, pressure, rest))
    v2 = jnp.where(interior, v*v, 0.)
    qfluid = (eps+pressure)/jnp.maximum(1-v2, 1e-8)
    penalty = jnp.mean(jax.nn.relu(jnp.where(candidate, v*v, 0.)-.95)**2)
    return eps, pressure, rest, qfluid, v, v2, h, interior, penalty


def source_terms(fields, alpha, matter, grid):
    """Physical-coordinate KEH sources (paper Eq. 24), retaining vacuum terms."""
    gamma, rho, omega = jnp.moveaxis(fields.value, -1, 0)
    r = fields.re*grid.s[:, None]/(1-grid.s[:, None])
    mu = grid.mu[None, :]
    dr = (1-grid.s[:, None])**2/fields.re
    first_r = fields.ds*dr[..., None]
    first_mu = fields.dm
    def dot(i, j):
        return first_r[..., i]*first_r[..., j]+(1-mu**2)/r**2*first_mu[..., i]*first_mu[..., j]
    def tee(i):
        return first_r[..., i]/r-mu/r**2*first_mu[..., i]
    eps, pressure, _, qfluid, _, v2, *_ = matter
    eg = jnp.exp(gamma/2)
    ea = jnp.exp(2*alpha)
    rotation = r*r*(1-mu*mu)*jnp.exp(-2*rho)*dot(2, 2)
    srho = eg*(8*jnp.pi*ea*qfluid*(1+v2)+rotation+tee(0)
               +rho*(8*jnp.pi*ea*pressure-.5*tee(0)-.25*dot(0, 0)))
    sgamma = eg*(16*jnp.pi*ea*pressure*(1+gamma/2)-gamma/4*dot(0, 0))
    somega = jnp.exp(gamma/2-rho)*(-16*jnp.pi*ea*qfluid*fields.omega
              +omega*(8*jnp.pi*ea*(eps+2*pressure)-rotation+dot(1, 1)
                      -.25*dot(0, 0)-2*tee(1)-.5*tee(0)))
    return jnp.stack([sgamma, srho, somega])


def green_map(sources, fields, grid):
    """Invert D4, D3, D5 on the full compactified spatial domain."""
    modes = jnp.einsum('cnijk,cjk->cni', grid.green, sources)
    inverse = jnp.einsum('cni,cnj->cij', modes, grid.angular)*fields.re**2
    gamma, rho, _ = jnp.moveaxis(fields.value, -1, 0)
    prefactor = jnp.stack([jnp.exp(-gamma/2), jnp.exp(-gamma/2), jnp.exp(rho-gamma/2)])
    return jnp.moveaxis(inverse*prefactor, 0, -1)


def residual_and_observables(fields, grid, eos):
    """Physics-only fixed-point residual, validity diagnostics and integral outputs."""
    alpha = alpha_from_fields(fields, grid)
    matter = matter_from_fields(fields, grid, eos)
    mapped = green_map(source_terms(fields, alpha, matter, grid), fields, grid)
    numerator = jnp.mean((mapped-fields.value)**2, axis=(0, 1))
    denominator = jnp.mean((mapped*mapped+fields.value*fields.value)/2, axis=(0, 1))
    denominator += jnp.array([1e-4, 1e-4, 1e-6+.01*fields.omega**2])
    residual = numerator/denominator
    eps, pressure, rest, qfluid, velocity, v2, _, interior, penalty = matter
    gamma, rho, omega = jnp.moveaxis(fields.value, -1, 0)
    r = fields.re*grid.s[:, None]/(1-grid.s[:, None])
    measure = (4*jnp.pi*jnp.exp(2*alpha+gamma)*r*r
               *fields.re/(1-grid.s[:, None])**2*grid.ws[:, None]*grid.wm[None, :])
    mass_integrand = (qfluid*(1+v2)+2*pressure
                      +2*omega*r*jnp.sqrt(1-grid.mu[None, :]**2)*jnp.exp(-rho)*qfluid*velocity)
    mass = jnp.sum(measure*mass_integrand)
    angular_momentum = jnp.sum(measure*qfluid*(fields.omega-omega)
                               *jnp.exp(-2*rho)*r*r*(1-grid.mu[None, :]**2))
    baryon_mass = jnp.sum(measure*rest*jnp.exp(-(gamma+rho)/2)
                         /jnp.sqrt(jnp.maximum(1-v2, 1e-8)))
    radius = fields.re*jnp.exp((fields.equator[0]-fields.equator[1])/2)
    loss = jnp.sum(residual)+100*(fields.validity_penalty+penalty)
    diagnostics = {
        'loss': loss, 'residual': residual, 'mass': mass, 'radius': radius,
        'baryon_mass': baryon_mass, 'angular_momentum': angular_momentum,
        'omega': fields.omega, 're': fields.re,
        'max_v2': jnp.max(jnp.where(interior, v2, 0.)),
        'validity_penalty': fields.validity_penalty+penalty,
        'matter_cells': jnp.sum(interior),
    }
    return loss, diagnostics
