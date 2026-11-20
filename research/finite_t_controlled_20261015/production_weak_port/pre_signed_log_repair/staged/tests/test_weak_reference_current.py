"""Forward current tests before thermal moments: independent traces and spin sums."""
import math

import numpy as np
import pytest

from nmbackend.reference import _weak_current as C
from nmbackend.reference import _weak_decay as D
from nmbackend.reference import _weak_two_body as B


def emission_event(channel, energy, cosine):
    mass, final = (C.MP, C.MN) if channel == 'EC' else (C.MN, C.MP)
    lepton_p = math.sqrt((energy-C.ME)*(energy+C.ME))
    parent = np.array([[mass, 0., 0., 0.]])
    lepton = np.array([[energy, 0., 0., lepton_p]])
    total = parent+lepton
    s = C.dot(total, total)
    root = np.sqrt(s)
    pf = (s-final*final)/(2*root)
    direction = np.array([[math.sqrt(1-cosine*cosine), 0., cosine]])
    beta = total[:, 1:]/total[:, 0, None]
    nu = C.boost(np.column_stack([pf, pf[:, None]*direction]), beta)
    daughter = C.boost(np.column_stack([(s+final*final)/(2*root), -pf[:, None]*direction]), beta)
    pn, pp = (daughter, parent) if channel == 'EC' else (parent, daughter)
    return nu, pn, lepton, pp


def slash(p):
    return sum(C.METRIC[i]*p[i]*C.G[i] for i in range(4))


def adj(m):
    return C.G[0]@m.conj().T@C.G[0]


def capture_trace(channel, k, n, l, p):
    anti = channel == 'PC'
    a, b, ma, mb = (p, n, C.MP, C.MN) if anti else (n, p, C.MN, C.MP)
    v = C.adjoint(C.vertices(-(k-l)))[0] if anti else C.vertices(k-l)[0]
    left = C.G@(C.IDENTITY-C.G5)
    h = np.array([[np.trace((slash(b[0])+mb*C.IDENTITY)@v[u]@(slash(a[0])+ma*C.IDENTITY)@adj(v[w])) for w in range(4)] for u in range(4)])
    if anti:
        lept = np.array([[np.trace(slash(k[0])@left[u]@(slash(l[0])-C.ME*C.IDENTITY)@adj(left[w])) for w in range(4)] for u in range(4)])
    else:
        lept = np.array([[np.trace((slash(l[0])+C.ME*C.IDENTITY)@left[u]@slash(k[0])@adj(left[w])) for w in range(4)] for u in range(4)])
    return C.GF**2*C.VUD**2/8*np.sum(C.METRIC[:, None]*C.METRIC[None, :]*lept*h)


@pytest.mark.parametrize('channel', ['EC', 'PC'])
@pytest.mark.parametrize('energy,cosine', [(2.37, -.63), (7.19, .21), (13.4, .87)])
def test_capture_literal_scalar_trace(channel, energy, cosine):
    vectors = emission_event(channel, energy, cosine)
    scalar = C.scalar_capture(channel, *vectors)[0]
    literal = C.literal_capture(channel, *vectors)[0]
    trace = capture_trace(channel, *vectors)
    assert abs(trace.imag) <= 3e-10*scalar
    assert abs(literal/scalar-1) <= 3e-10
    assert abs(trace.real/scalar-1) <= 3e-10
    k, n, l, p = vectors
    conservation = p+l-n-k if channel == 'EC' else n+l-p-k
    assert np.max(abs(conservation)) < 2e-10*C.MN


def test_decay_reciprocal_and_transfer():
    n, p, l, k, _, _ = D.pair_events(np.array([.37, .71, 1.12]), np.array([-.77, .19, .83]))
    f = C.decay_amplitude(n, p, l, k)
    b = C.decay_amplitude(n, p, l, k, inverse=True)
    assert np.max(abs(f/b-1)) < 3e-10
    assert np.max(abs((p-n)+(l+k))) < 2e-10*C.MN
    assert np.all((p-n)[:, 0] < 0)
    assert np.all(C.dot(l+k, l+k) >= C.ME*C.ME)


def test_reciprocal_adjoint_and_form_factor_limits():
    q = np.array([[-1.7, .3, .4, 4.2], [-.8, .1, .2, .4]])
    vertex = C.vertices(q)
    assert np.max(abs(C.adjoint(C.adjoint(vertex))-vertex)) < 1e-13
    gv, ga, f, gp = C.form_factors(np.array([0.]))
    assert gv[0] == 1 and ga[0] == C.GA and f[0] == 3.706
    assert gp[0] == pytest.approx(2*C.MW**2*C.GA/139.57**2, rel=2e-15)
    assert not C.G.flags.writeable and not C.G5.flags.writeable


def test_exact_moving_electron_threshold_and_rest_antinu_threshold():
    ep = math.hypot(C.MP, 8.1)
    cosine = np.array([-.9, .13, .8])
    el = B.electron_threshold(ep, 8.1, cosine)
    pe = np.sqrt((el-C.ME)*(el+C.ME))
    s = C.MP**2+C.ME**2+2*(ep*el-8.1*pe*cosine)
    assert np.max(abs(s/C.MN**2-1)) < 2e-14
    threshold = ((C.MN+C.ME)**2-C.MP*C.MP)/(2*C.MP)
    assert abs(threshold-1.8060662080187608) < 3e-12


def test_zero_transfer_point_and_magnetic_transversality():
    zero = np.zeros((1, 4))
    assert np.array_equal(C.vertices(zero), C.vertices(zero, point=True))
    q = np.array([[-1.3, .4, -.2, 2.7], [-.7, .12, .19, .3]])
    t = C.dot(q, q)
    v, a, _, gp = C.form_factors(t)
    va = v[:, None, None, None]*C.G[None]-a[:, None, None, None]*(C.G@C.G5)[None]
    pseudo = -gp[:, None, None, None]/C.MW*q[:, :, None, None]*C.G5[None, None]
    magnetic = C.vertices(q)-va-pseudo
    contraction = np.einsum('nu,u,nuab->nab', q, C.METRIC, magnetic)
    assert np.max(abs(contraction)) < 2e-13
    gp_contract = np.einsum('nu,u,nuab->nab', q, C.METRIC, pseudo)
    expected = -gp[:, None, None]*t[:, None, None]/C.MW*C.G5[None]
    assert np.max(abs(gp_contract-expected)) < 2e-13
