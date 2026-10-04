"""Independent contact roots, central geometry, GR volumes and moving interfaces."""
from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np

from nmbackend.eos.base import EOSTable
from nmbackend.tov.solver import TOVProfile
from nmbackend.cooling.physics.geometry import (
    enthalpy_at_energy, geometry_at_volume, volume_at_h, volume_quadrature,
)


def quadratic_table(a=3.):
    h = jnp.array([0., .2])
    p = a*h*h+h+.1
    eps = 2*a*h+1-p
    return EOSTable(jnp.exp(h), eps, p, h, jnp.ones(2)*.1, eps, p)


def profile(mass=.2, radius=1.):
    r = jnp.array([.1, .4, .7, 1.])*radius
    h = .2-.2*(r/radius)**2
    m = mass*(r/radius)**3
    phi = jnp.sqrt(1-2*mass/radius)*jnp.exp(-h)
    return TOVProfile(h, r, m, phi)


def exact_volume(r, mass=.2, radius=1.):
    k = 2*mass/radius**3
    u = np.sqrt(k)*r
    return 2*np.pi/k**1.5*(np.arcsin(u)-u*np.sqrt(1-u*u))


def test_density_contact_against_analytic_root_and_endpoint_sensitivities():
    f = lambda x: enthalpy_at_energy(quadratic_table(x[0]), x[1])
    x = jnp.array([3., 1.5])
    root = (5-np.sqrt(25-12*(1.5-.9)))/6
    expected = np.array([-(2*root-root*root)/(5-6*root), 1/(5-6*root)])
    np.testing.assert_allclose(jax.jit(f)(x), root, atol=1e-14)
    np.testing.assert_allclose(jax.jacfwd(f)(x), expected, rtol=1e-13)
    np.testing.assert_allclose(jax.jacrev(f)(x), expected, rtol=1e-13)
    for step in (1e-4, 3e-5, 1e-5):
        fd = np.array([float((f(x+jnp.eye(2)[k]*step)-f(x-jnp.eye(2)[k]*step))/(2*step))
                       for k in range(2)])
        np.testing.assert_allclose(fd, expected, rtol=3e-8, atol=1e-10)

    # Knot support is sensitive to roundoff in recomputing Hermite endpoints.
    h = jnp.array([0., .1, .2])
    p, e = jnp.array([.1, .22, .398]), jnp.array([.9, 1.2, 1.78])
    knot = EOSTable(jnp.ones(3), e, p, h, jnp.ones(3)*.1, e, p)
    np.testing.assert_allclose(jax.vmap(lambda e: enthalpy_at_energy(knot, e))(e), h, atol=1e-14)
    np.testing.assert_allclose(jax.grad(lambda e: enthalpy_at_energy(knot, e))(1.2), 1/3.38, rtol=1e-13)


def test_invalid_or_unsupported_density_roots_are_rejected():
    t = quadratic_table()
    for energy in (.8, 2., np.nan):
        assert np.isnan(enthalpy_at_energy(t, energy))
    for invalid in (t._replace(h=t.h.at[1].set(0.)),
                    t._replace(eps=t.eps.at[1].set(t.eps[0])),
                    t._replace(p=t.p.at[0].set(jnp.nan))):
        assert np.isnan(enthalpy_at_energy(invalid, 1.5))

    # Ordered positive nodes do not ensure monotonic density between them.
    bad = t._replace(eps=jnp.array([1., 2.]), p=jnp.array([.1, .4]))
    assert np.all(np.diff(bad.eps) > 0) and np.all(bad.p > 0)
    assert np.isnan(enthalpy_at_energy(bad, 1.5))


def test_central_geometry_preserves_radius_mass_and_exact_lapse():
    s = profile()
    v = jnp.array([0., 1e-9, .03, .4, 1.])
    g = jax.jit(geometry_at_volume)(s, .2, v)
    r = np.cbrt(v)
    np.testing.assert_allclose(g.r_km, r, rtol=1e-14, atol=1e-16)
    np.testing.assert_allclose(g.m_km, .2*np.asarray(v), rtol=1e-14, atol=0.)
    np.testing.assert_allclose(g.h, .2-.2*r*r, atol=1e-15)
    np.testing.assert_allclose(g.ephi, np.sqrt(.6)*np.exp(-np.asarray(g.h)), rtol=1e-14)
    h = .2-1e-6
    expected_v = ((.2-h)/.2)**1.5
    np.testing.assert_allclose(volume_at_h(s, .2, h), expected_v, rtol=1e-12)
    np.testing.assert_allclose(jax.grad(lambda h: volume_at_h(s, .2, h))(h),
                               -1.5/.2*np.sqrt((.2-h)/.2), rtol=1e-12)
    np.testing.assert_allclose(jax.grad(lambda v: geometry_at_volume(s, .2, v).m_km)(1e-9), .2)


def test_contact_quadrature_against_gr_volume_and_moving_jump():
    def f(x, order):
        mass, radius, contact, inner, outer = x
        _, weights = volume_quadrature(profile(mass, radius), .2,
                                       jnp.array([0., contact**3, radius**3]), order=order)
        return jnp.sum(weights*jnp.array([inner, outer])[:, None])/1e15
    x = jnp.array([.2, 1., .6, 3., 1.])
    va, vr = exact_volume(.6), exact_volume(1.)
    exact = 3*va+vr-va
    expected = np.array([4*np.pi*.6**2*2/np.sqrt(1-.4*.6**2), va, vr-va])
    errors = []
    for order in (16, 32, 64, 128):
        fn = lambda x: f(x, order)
        forward = np.asarray(jax.jacfwd(fn)(x))
        np.testing.assert_allclose(jax.jacrev(fn)(x), forward, rtol=2e-13, atol=1e-13)
        errors.append(abs(float(fn(x))/exact-1))
    assert np.all(np.diff(errors) < 0)
    np.testing.assert_allclose(f(x, 128), exact, rtol=2e-9)
    np.testing.assert_allclose(forward[[2, 3, 4]], expected, rtol=2e-9)
    np.testing.assert_allclose(f(x.at[3:].set(1.), 128), vr, rtol=2e-9)
    for step in (1e-4, 3e-5, 1e-5):
        d = jnp.zeros(5).at[2].set(step)
        np.testing.assert_allclose((f(x+d, 128)-f(x-d, 128))/(2*step), expected[0], rtol=3e-8)


def test_zero_width_segments_have_finite_geometry_and_zero_weights():
    def f(a):
        g, weights = volume_quadrature(profile(), .2, jnp.array([0., 0., a**3, a**3, 1.]))
        return g, weights
    g, weights = f(.6)
    assert np.all(np.isfinite(np.asarray(g)))
    np.testing.assert_array_equal(np.asarray(weights)[[0, 2]], 0.)
    assert np.all(np.isfinite(jax.jacfwd(lambda a: f(a)[1])(.6)))


def test_stellar_coverage_is_distinct_from_eos_coverage():
    s = profile()
    for h in (-.01, .201, np.nan):
        assert np.isnan(volume_at_h(s, .2, h))
    for v in (-.01, 1.001, np.nan):
        assert np.all(np.isnan(np.asarray(geometry_at_volume(s, .2, v))))
    for edges in ([0., .8, .2, 1.], [-.001, 1.], [0., 1.001]):
        _, weights = volume_quadrature(s, .2, jnp.array(edges))
        assert np.all(np.isnan(weights))
    invalid = s._replace(ephi=s.ephi.at[-1].set(jnp.inf))
    assert np.all(np.isnan(np.asarray(geometry_at_volume(invalid, .2, .1))))


def test_physical_eos_contacts_at_bisection_precision_limit():
    from nmbackend.constants import MEVFM3_TO_GEOM
    from nmbackend.eos import MetaModelEOS
    from nmbackend.eos.crust import _G_CM3_TO_MEVFM3

    # Supported monotone intervals whose midpoint residual can miss the tolerance.
    rho = jnp.array([557028487e6, 502078121e6, 360750247e6, 355662730e6,
                     315644319e6, 260351642e6, 253420918e6, 241418729e6, 205050564e6])
    model = MetaModelEOS(muons=True, s_high='power', n_points_low=240, n_points_tail=160)
    table = model.eos_table({'L_sym': 100., 'K_sym': -100.})
    energy = rho * _G_CM3_TO_MEVFM3 * MEVFM3_TO_GEOM
    roots = jax.jit(jax.vmap(lambda e: enthalpy_at_energy(table, e)))(energy)
    assert np.all(np.isfinite(roots))
    recovered = jax.vmap(lambda h: table.at_h(h)[0])(roots)
    # Separate polynomial evaluation has its own rounding budget.
    np.testing.assert_allclose(recovered, energy, rtol=1e-13, atol=0.)


def test_live_eos_contacts_retain_compiled_ad_support():
    from nmbackend.constants import MEVFM3_TO_GEOM
    from nmbackend.eos import MetaModelEOS
    from nmbackend.eos.crust import _G_CM3_TO_MEVFM3

    # Own-drip and chemistry contacts that previously failed only with live EOS nodes.
    energy = jnp.array([430000013312., 1.334e14])*_G_CM3_TO_MEVFM3*MEVFM3_TO_GEOM
    model = MetaModelEOS(muons=True, s_high='power', n_points_low=240, n_points_tail=160)

    @jax.jit
    def contacts(L):
        table = model.eos_table({'L_sym': L, 'K_sym': -100.})
        return jax.vmap(lambda e: enthalpy_at_energy(table, e))(energy)

    forward, tangent = jax.jvp(contacts, (100.,), (1.,))
    reverse, pullback = jax.vjp(contacts, 100.)
    assert np.all(np.isfinite(forward)) and np.all(np.isfinite(reverse))
    assert np.all(np.isfinite(tangent))
    np.testing.assert_allclose(reverse, forward, rtol=1e-13, atol=1e-16)
    np.testing.assert_allclose(pullback(jnp.ones_like(reverse))[0], jnp.sum(tangent),
                               rtol=1e-12, atol=1e-16)
    for step in (.04, .02, .01):
        fd = (contacts(100.+step)-contacts(100.-step))/(2*step)
        np.testing.assert_allclose(tangent, fd, rtol=1e-5, atol=1e-11)
