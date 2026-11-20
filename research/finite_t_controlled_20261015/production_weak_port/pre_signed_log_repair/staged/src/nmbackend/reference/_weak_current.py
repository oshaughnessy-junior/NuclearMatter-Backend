"""Fixed free-nucleon Guo full tree current, with explicit reciprocal crossing.

Source: arxiv:2006.12051v2 Eqs26–29,47; no measured-lifetime calibration.
"""
from __future__ import annotations

import numpy as np

MN = 939.56542052
MP = 938.27208816
ME = .51099895069
MW = (MN+MP)/2
GF = 1.1663785e-11
VUD = .974
HBAR = 6.582119569e-22
HC = 197.3269804
GA = 1.2723
SOURCE = 'Guo2020-v2-full-free-VA-F2-GP-MV840-MA1000-mpi139.57'
SIGMA = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], complex)
ZERO = np.zeros((2, 2), complex)
G = np.array([np.diag([1., 1., -1., -1.])]+[np.block([[ZERO, s], [-s, ZERO]]) for s in SIGMA])
G5 = 1j*G[0]@G[1]@G[2]@G[3]
IDENTITY = np.eye(4)
METRIC = np.array([1., -1., -1., -1.])
COMM = np.array([[G[u]@G[v]-G[v]@G[u] for v in range(4)] for u in range(4)])
for _array in (SIGMA, ZERO, G, G5, IDENTITY, METRIC, COMM):
    _array.setflags(write=False)
del _array


def dot(a, b):
    return a[..., 0]*b[..., 0]-np.sum(a[..., 1:]*b[..., 1:], axis=-1)


def boost(v, beta):
    b2 = np.sum(beta*beta, axis=1)
    if np.any(b2 >= 1) or not np.all(np.isfinite(b2)):
        raise ArithmeticError('Nonphysical event boost')
    gamma = 1/np.sqrt(1-b2)
    bv = np.sum(beta*v[:, 1:], axis=1)
    factor = gamma*v[:, 0]+gamma**2/(gamma+1)*bv
    return np.column_stack([gamma*(v[:, 0]+bv), v[:, 1:]+factor[:, None]*beta])


def form_factors(t):
    if not np.all(np.isfinite(t)) or np.any(t >= min(139.57**2, 840.**2, 1000.**2, 4*MW*MW)):
        raise ArithmeticError('Outside sub-pole free current domain')
    den = (1-t/(4*MW*MW))*(1-t/840.**2)**2
    return ((1-4.706*t/(4*MW*MW))/den, GA/(1-t/1000.**2)**2,
            3.706/den, 2*MW*MW*GA/((1-t/1000.**2)**2*(139.57**2-t)))


def vertices(q_gamma, point=False):
    """Gamma_np with q_Gamma=pp−pn; GP and F2 linear signs retained."""
    t = dot(q_gamma, q_gamma)
    if point:
        return np.broadcast_to(G-GA*(G@G5), (len(q_gamma), 4, 4, 4))
    v, a, f, gp = form_factors(t)
    return (v[:, None, None, None]*G[None]-a[:, None, None, None]*(G@G5)[None]
            - f[:, None, None, None]/(4*MW)*np.einsum('nj,j,ujab->nuab', q_gamma, METRIC, COMM)
            - gp[:, None, None, None]/MW*q_gamma[:, :, None, None]*G5[None, None])


def adjoint(v):
    return np.einsum('ab,nubc,cd->nuad', G[0], v.conj().swapaxes(-1, -2), G[0])


def spinors(p, mass, anti=False):
    root = np.sqrt(p[:, 0]+mass)
    sigma_p = np.einsum('ni,ijk->njk', p[:, 1:], SIGMA)
    large = root[:, None, None]*np.eye(2)
    small = sigma_p/root[:, None, None]
    return np.concatenate([small, large] if anti else [large, small], axis=1)


def literal_capture(channel, k, pn, lepton, pp):
    """Explicit external u/v sums, averaged FOUR forward baryon/lepton spins."""
    anti = channel == 'PC'
    if channel not in ('EC', 'PC'):
        raise ValueError('Named capture channel required')
    pa, pb, ma, mb = (pp, pn, MP, MN) if anti else (pn, pp, MN, MP)
    ua, ub = spinors(pa, ma), spinors(pb, mb)
    uk, ul = spinors(k, 0., anti), spinors(lepton, ME, anti)
    q_abs = k-lepton
    vertex = adjoint(vertices(-q_abs)) if anti else vertices(q_abs)
    left = G@(IDENTITY-G5)
    baryon = np.einsum('nai,ab,nubc,ncj->nuij', ub.conj(), G[0], vertex, ua).reshape((-1, 4, 4))
    if anti:
        lep = np.einsum('nai,ab,ubc,ncj->nuij', uk.conj(), G[0], left, ul).reshape((-1, 4, 4))
    else:
        lep = np.einsum('nai,ab,ubc,ncj->nuij', ul.conj(), G[0], left, uk).reshape((-1, 4, 4))
    amp = np.einsum('nui,u,nuj->nij', lep, METRIC, baryon)
    return GF**2*VUD**2/8*np.sum(abs(amp)**2, axis=(1, 2))


def scalar_capture(channel, k, pn, lepton, pp):
    """Guo47 invariant reduction, same avg4 as literal_capture, no normalization fit.

    Port of author EC invariant reduction and primary Guo crossed sign, not an
    import/copy of the independent reviewer's quadrature or runtime reference.
    """
    if channel not in ('EC', 'PC'):
        raise ValueError('Named capture channel required')
    a, b, ma, mb, sign = (pn, pp, MN, MP, 1) if channel == 'EC' else (pp, pn, MP, MN, -1)
    q = b-a
    t = dot(q, q)
    v, axial, f, gp = form_factors(t)
    kn, kb, ln, lb, kl, ab = [dot(x, y) for x, y in ((k, a), (k, b), (lepton, a), (lepton, b), (k, lepton), (a, b))]
    kq, lq, aq, bq = [dot(x, q) for x in (k, lepton, a, b)]
    vv = v*v*(kn*lb+kb*ln-ma*mb*kl)
    va = sign*2*v*axial*(kn*lb-kb*ln)
    aa = axial*axial*(kn*lb+kb*ln+ma*mb*kl)
    vf = v*f/(2*MW)*((kn*mb-kb*ma)*lq+(ln*mb-lb*ma)*kq+(aq*mb-bq*ma)*kl)
    af = sign*axial*f/MW*((kn*mb+kb*ma)*lq-(ln*mb+lb*ma)*kq)
    ff = f*f/(8*MW*MW)*(2*(ln*bq+lb*aq)*kq+2*(kn*bq+kb*aq)*lq-2*kq*lq*ab+t*(kl*ab-2*kn*lb-2*kb*ln)-ma*mb*(kl*t+2*kq*lq))
    ap = axial*gp/MW*(kq*(ln*mb-lb*ma)+lq*(kn*mb-kb*ma)+kl*(bq*ma-aq*mb))
    gg = gp*gp/(2*MW*MW)*(2*kq*lq-t*kl)*(ab-ma*mb)
    amplitude = 8*GF**2*VUD**2*(vv+va+aa+vf+af+ff+ap+gg)
    if not np.all(np.isfinite(amplitude)) or np.any(amplitude <= 0):
        raise ArithmeticError('Positive full-current contraction unresolved')
    return amplitude


def decay_amplitude(pn, pp, electron, anti, inverse=False, point=False):
    """Neutron-spin avg2. Decay future Q=electron+anti=−q_Gamma."""
    vertex = vertices(pp-pn, point=point)
    un, up = spinors(pn, MN), spinors(pp, MP)
    ue, va = spinors(electron, ME), spinors(anti, 0., True)
    left = G@(IDENTITY-G5)
    if inverse:
        baryon = np.einsum('nai,ab,nubc,ncj->nuij', un.conj(), G[0], adjoint(vertex), up).reshape((-1, 4, 4))
        lep = np.einsum('nai,ab,ubc,ncj->nuij', va.conj(), G[0], left, ue).reshape((-1, 4, 4))
    else:
        baryon = np.einsum('nai,ab,nubc,ncj->nuij', up.conj(), G[0], vertex, un).reshape((-1, 4, 4))
        lep = np.einsum('nai,ab,ubc,ncj->nuij', ue.conj(), G[0], left, va).reshape((-1, 4, 4))
    amp = np.einsum('nui,u,nuj->nij', lep, METRIC, baryon)
    result = GF**2*VUD**2/4*np.sum(abs(amp)**2, axis=(1, 2))
    if not np.all(np.isfinite(result)) or np.any(result <= 0):
        raise ArithmeticError('Positive decay current contraction unresolved')
    return result
