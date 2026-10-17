# ASTselected puredefinitions, injectedglobalsinprimitives.py; no upstreamdriver
GF = 1.1663785e-11
VUD = 0.974
HBAR = 6.582119569e-22
HC = 197.3269804
S = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], complex)
Z = np.zeros((2, 2), complex)
G = np.array([np.diag([1.0, 1.0, -1.0, -1.0])] + [np.block([[Z, s], [-s, Z]]) for s in S])
G5 = 1j * G[0] @ G[1] @ G[2] @ G[3]
I = np.eye(4)
MET = np.array([1.0, -1.0, -1.0, -1.0])

def spinors(p, m, anti=False):
    root = np.sqrt(p[:, 0] + m)
    sig = np.einsum('ni,ijk->njk', p[:, 1:], S)
    big = root[:, None, None] * np.eye(2)
    small = sig / root[:, None, None]
    return np.concatenate([small, big] if anti else [big, small], axis=1)

def adj(v):
    return np.einsum('ab,nubc,cd->nuad', G[0], v.conj().swapaxes(-1, -2), G[0])

def vertices(q, M):
    t = q[:, 0] ** 2 - np.sum(q[:, 1:] ** 2, axis=1)
    if np.any(t >= min(139.57 ** 2, 840.0 ** 2, 1000.0 ** 2, 4 * M * M)):
        raise ValueError('Sub-pole current required')
    den = (1 - t / (4 * M * M)) * (1 - t / 840.0 ** 2) ** 2
    v = (1 - 4.706 * t / (4 * M * M)) / den
    a = 1.2723 / (1 - t / 1000.0 ** 2) ** 2
    f = 3.706 / den
    gp = 2 * M * M * a / (139.57 ** 2 - t)
    comm = np.array([[G[u] @ G[j] - G[j] @ G[u] for j in range(4)] for u in range(4)])
    return v[:, None, None, None] * G[None, :, :, :] - a[:, None, None, None] * (G @ G5)[None, :, :, :] - f[:, None, None, None] / (4 * M) * np.einsum('nj,j,ujab->nuab', q, MET, comm) - gp[:, None, None, None] / M * q[:, :, None, None] * G5[None, None, :, :]

def boost(v, beta):
    b2 = np.sum(beta * beta, axis=1)
    gamma = 1 / np.sqrt(1 - b2)
    dot = np.sum(beta * v[:, 1:], axis=1)
    E = gamma * (v[:, 0] + dot)
    factor = gamma * v[:, 0] + gamma ** 2 / (gamma + 1) * dot
    return np.column_stack([E, v[:, 1:] + factor[:, None] * beta])
