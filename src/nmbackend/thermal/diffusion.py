"""Conservative scalar transport on a frozen axisymmetric metric.

Discretizes Beznogov et al. (2023), Eq. 10, in (r, mu=cos(theta)).
Closed rectangular meridional domains only: no envelope or oblate surface.
All inputs use a single consistent unit system; no physical constants implied.
"""
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg


class ThermalOperator(NamedTuple):
    """Cell capacity and internal face conductances, with insulated boundaries."""

    capacity: jnp.ndarray
    radial: jnp.ndarray
    angular: jnp.ndarray
    source_weight: jnp.ndarray


def metric_factors(rho, gama, alpha, velocity_sq):
    """RNS potentials -> lapse N, meridional A, azimuthal B, Lorentz Gamma.

    velocity_sq must be the physical ZAMO velocity squared divided by c^2,
    restricted to stellar matter with 0 <= velocity_sq < 1.
    """
    return (jnp.exp((gama + rho) / 2), jnp.exp(alpha),
            jnp.exp((gama - rho) / 2), 1 / jnp.sqrt(1 - velocity_sq))


def build_operator(r_faces, mu_faces, lapse, meridional, azimuthal,
                   lorentz, heat_capacity, conductivity):
    """Midpoint finite volumes for positive, temperature-independent Cv and K.

    Fields broadcast to (nr, nmu). All fields must be positive; dt >= 0.
    r_faces >= 0 and mu_faces in [-1,1]
    must increase. Faces use distance-weighted series resistance. Geometry is fixed
    during differentiation; this does not differentiate an RNS solution.
    """
    r_faces, mu_faces = jnp.asarray(r_faces), jnp.asarray(mu_faces)
    r = (r_faces[:-1] + r_faces[1:]) / 2
    mu = (mu_faces[:-1] + mu_faces[1:]) / 2
    dr, dm = jnp.diff(r_faces), jnp.diff(mu_faces)
    shape = (r.size, mu.size)
    N, A, B, G, cv, k = [jnp.broadcast_to(jnp.asarray(x), shape) for x in
                         (lapse, meridional, azimuthal, lorentz,
                          heat_capacity, conductivity)]
    volume = (jnp.diff(r_faces**3) / 3)[:, None] * dm[None, :]
    capacity = volume * A**2 * B * G * cv
    knb = k * N * B
    radial = (r_faces[1:-1, None]**2 * dm[None, :] /
              (dr[:-1, None]/(2*knb[:-1]) + dr[1:, None]/(2*knb[1:])))
    angular = (dr[:, None] * (1-mu_faces[None, 1:-1]**2) /
               (dm[None, :-1]/(2*knb[:, :-1]) + dm[None, 1:]/(2*knb[:, 1:])))
    return ThermalOperator(capacity, radial, angular,
                           volume * A**2 * B * N**2 / G)


def diffusion(op, temperature):
    """Integrated divergence; constant redshifted temperature is exactly null."""
    radial = op.radial * jnp.diff(temperature, axis=0)
    angular = op.angular * jnp.diff(temperature, axis=1)
    return (jnp.pad(radial, ((0, 1), (0, 0)))
            - jnp.pad(radial, ((1, 0), (0, 0)))
            + jnp.pad(angular, ((0, 0), (0, 1)))
            - jnp.pad(angular, ((0, 0), (1, 0))))


def implicit_step(op, redshifted_temperature, dt, source=0., tol=1e-11,
                  maxiter=2000):
    """Backward Euler solve, including a prescribed local heating Q (negative loss).

    Uses capacity scaling for an SPD system. JAX CG differentiates the solve
    implicitly; callers must check the returned relative residual. source
    is not evaluated implicitly as a temperature-dependent emissivity.
    """
    root = jnp.sqrt(op.capacity)
    apply = lambda z: z - dt * diffusion(op, z / root) / root
    rhs = root * redshifted_temperature + dt * op.source_weight * source / root
    z, _ = cg(apply, rhs, tol=tol, atol=0., maxiter=maxiter)
    residual = jnp.linalg.norm(apply(z) - rhs) / jnp.maximum(
        jnp.linalg.norm(rhs), jnp.finfo(rhs.dtype).tiny)
    return z / root, residual
