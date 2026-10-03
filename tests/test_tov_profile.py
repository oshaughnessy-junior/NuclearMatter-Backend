"""Native profiles against the exact constant-density Schwarzschild interior."""
import jax
import jax.numpy as jnp
import numpy as np
from nmbackend.eos.base import EOSTable
from nmbackend.tov.solver import solve_profile


def incompressible_table():
    h = jnp.linspace(1e-10, 1., 2049)
    eps = jnp.full_like(h, 1e-3)
    p = eps * jnp.expm1(h)
    # Structure-only: sound speed is unused; constant n is consistent with mu.
    return EOSTable(jnp.ones_like(h), eps, p, h, jnp.ones_like(h), eps, p)


def exact_radius(hc):
    x = jnp.expm1(hc)
    surface_redshift = (1. + x) / (1. + 3. * x)
    compactness = (1. - surface_redshift**2) / 2.
    return jnp.sqrt(3. * compactness / (4. * jnp.pi * 1e-3))


def test_profile_matches_schwarzschild_interior():
    profile = solve_profile(incompressible_table(), .3, rtol=1e-9, atol=1e-11)
    r, m, ephi = map(np.asarray, (profile.r_km, profile.m_km, profile.ephi))
    R = float(exact_radius(.3))
    C = 4. * np.pi * 1e-3 * R**2 / 3.
    expected = (3. * np.sqrt(1.-2.*C) - np.sqrt(1.-2.*C*(r/R)**2)) / 2.
    np.testing.assert_allclose(r[-1], R, rtol=2e-6)
    np.testing.assert_allclose(m, 4.*np.pi*1e-3*r**3/3., rtol=2e-6, atol=1e-10)
    np.testing.assert_allclose(ephi, expected, rtol=2e-6)
    assert np.all(np.diff(r)>0) and np.all(np.diff(np.asarray(profile.h))<0)


def test_profile_central_enthalpy_sensitivity():
    table = incompressible_table()
    radius = lambda hc: solve_profile(table, hc, n_points=32, rtol=1e-9, atol=1e-11).r_km[-1]
    # Exact analytic forward value gates the derivative comparison.
    np.testing.assert_allclose(radius(.3), exact_radius(.3), rtol=2e-6)
    np.testing.assert_allclose(jax.jacfwd(radius)(.3), jax.jacfwd(exact_radius)(.3), rtol=2e-5)
