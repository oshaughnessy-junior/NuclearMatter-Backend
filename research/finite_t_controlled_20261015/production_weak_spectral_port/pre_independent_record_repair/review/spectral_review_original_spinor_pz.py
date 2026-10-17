# ASTselected puredefinitions, injectedglobalsinprimitives.py; no upstreamdriver
GF = 1.1663785e-11
VUD = 0.974
HBAR = 6.582119569e-22
S = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]], complex), np.diag([1.0, -1.0])]
Z = np.zeros((2, 2), complex)
G = [np.diag([1.0, 1.0, -1.0, -1.0]).astype(complex)] + [np.block([[Z, s], [-s, Z]]) for s in S]
G5 = 1j * G[0] @ G[1] @ G[2] @ G[3]
I = np.eye(4)
MET = np.array([1.0, -1.0, -1.0, -1.0])

@dataclass(frozen=True)
class Point:
    T: float
    ma: float
    mb: float
    mua: float
    mub: float
    Enu: float
    El: float
    cosine: float
    ml: float
    mul: float
    anti: bool = False
    Mweak: float = 938.91875434

    def __post_init__(self):
        if not all((math.isfinite(x) for k, x in vars(self).items() if k != 'anti')) or min(self.T, self.ma, self.mb, self.Enu, self.Mweak) <= 0 or self.ml < 0 or (self.El <= self.ml) or (not -1 < self.cosine < 1) or (not isinstance(self.anti, bool)):
            raise ValueError('Finite positive spacelike interior point required')

def adj(A):
    return G[0] @ A.conj().T @ G[0]

def spinors(p, m, anti=False):
    root = math.sqrt(p[0] + m)
    sig = sum((p[i + 1] * S[i] for i in range(3)))
    chi = np.eye(2)
    big = root * chi
    small = sig @ chi / root
    return np.vstack([small, big] if anti else [big, small])

def currents(q, M):
    t = q[0] ** 2 - q[1:] @ q[1:]
    if t >= min(139.57 ** 2, 840.0 ** 2, 1000.0 ** 2, 4 * M * M):
        raise ValueError('Sub-pole current required')
    den = (1 - t / (4 * M * M)) * (1 - t / 840.0 ** 2) ** 2
    v = (1 - 4.706 * t / (4 * M * M)) / den
    a = 1.2723 / (1 - t / 1000.0 ** 2) ** 2
    f = 3.706 / den
    gp = 2 * M * M * a / (139.57 ** 2 - t)
    return np.array([v * G[u] - a * G[u] @ G5 - f / (4 * M) * sum(((G[u] @ G[j] - G[j] @ G[u]) * MET[j] * q[j] for j in range(4))) - gp / M * q[u] * G5 for u in range(4)])

def setup(p, current_model='full'):
    if current_model not in ('full', 'point_VA'):
        raise ValueError('Unknown current model')
    pl = math.sqrt((p.El - p.ml) * (p.El + p.ml))
    q = math.sqrt(p.Enu ** 2 + pl ** 2 - 2 * p.Enu * pl * p.cosine)
    w = p.Enu - p.El
    a = q * q - w * w
    if a <= 0:
        raise ValueError('Only strictly spacelike kinetic transfer qualified')
    kz = (p.Enu ** 2 + q * q - pl * pl) / (2 * q)
    kx = math.sqrt(p.Enu ** 2 - kz * kz)
    k = np.array([p.Enu, kx, 0, kz])
    l = np.array([p.El, kx, 0, kz - q])
    transfer = k - l
    V = currents(transfer, p.Mweak) if not p.anti else np.array([adj(x) for x in currents(-transfer, p.Mweak)])
    if current_model == 'point_VA':
        V = np.array([G[u] - 1.2723 * G[u] @ G5 for u in range(4)])
    uk = spinors(k, 0, p.anti)
    ul = spinors(l, p.ml, p.anti)
    A = np.array([(uk.conj().T @ G[0] @ G[u] @ (I - G5) @ ul if p.anti else ul.conj().T @ G[0] @ G[u] @ (I - G5) @ uk).reshape(-1) for u in range(4)])
    L = A @ A.conj().T
    D = p.mb * p.mb - p.ma * p.ma + a
    return (pl, q, w, a, D, V, L, k, l)

def shell(p, u, phi):
    pl, q, w, a, D, V, L, k, l = setup(p)
    r = math.sqrt(2 * p.ma * u)
    z0 = -D / (2 * math.sqrt(a))
    ep = math.sqrt(p.ma * p.ma + r * r + z0 * z0)
    z = (q * z0 + w * ep) / math.sqrt(a)
    Ea = (q * ep + w * z0) / math.sqrt(a)
    Eb = Ea + w
    pa = np.array([Ea, r * math.cos(phi), r * math.sin(phi), z])
    pb = pa + np.array([w, 0, 0, q])
    jac = abs((z + q) / Eb - z / Ea)
    return (pa, pb, jac)

def integrate(p, order=24, nphi=8, tail=40.0, current_model='full'):
    if isinstance(order, bool) or not isinstance(order, int) or order < 4 or isinstance(nphi, bool) or (not isinstance(nphi, int)) or (nphi < 4) or (not math.isfinite(tail)) or (not 12 <= tail <= 120):
        raise ValueError('Integer orders>=4 and finite tail12..120 required')
    pl, q, w, a, D, V, L, k, l = setup(p, current_model)
    z0 = -D / (2 * math.sqrt(a))
    E0 = (q * math.sqrt(p.ma * p.ma + z0 * z0) + w * z0) / math.sqrt(a)
    Emax = max(E0, p.mua, p.mub - w) + tail * p.T

    def u_at(E):
        return (((E * math.sqrt(a) - w * z0) / q) ** 2 - p.ma * p.ma - z0 * z0) / (2 * p.ma)
    umax = u_at(Emax)
    cuts = [0.0, umax]
    for center in [p.mua, p.mub - w]:
        for h in [-16, -8, -3, 0, 3, 8, 16]:
            E = center + h * p.T
            if E0 < E < Emax:
                cuts.append(u_at(E))
    cuts = sorted(set(cuts))
    x, g = leggauss(order)
    W = np.zeros((2, 4, 4), complex)
    maxres = 0.0
    minjac = math.inf
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        for t, wt in zip((lo + hi) / 2 + (hi - lo) * x / 2, g * (hi - lo) / 2):
            r = math.sqrt(2 * p.ma * t)
            ep = math.sqrt(p.ma * p.ma + r * r + z0 * z0)
            z = (q * z0 + w * ep) / math.sqrt(a)
            Ea = (q * ep + w * z0) / math.sqrt(a)
            Eb = Ea + w
            jac = abs((z + q) / Eb - z / Ea)
            minjac = min(minjac, jac)
            occ = np.array([expit((p.mua - Ea) / p.T) * expit((Eb - p.mub) / p.T), expit((p.mub - Eb) / p.T) * expit((Ea - p.mua) / p.T)])
            for phi in 2 * math.pi * np.arange(nphi) / nphi:
                pa = np.array([Ea, r * math.cos(phi), r * math.sin(phi), z])
                pb = pa + np.array([w, 0, 0, q])
                ua = spinors(pa, p.ma)
                ub = spinors(pb, p.mb)
                J = np.array([(ub.conj().T @ G[0] @ v @ ua).reshape(-1) for v in V])
                H = J @ J.conj().T
                W += occ[:, None, None] * H[None, :, :] * p.ma * wt * (2 * math.pi / nphi) / (16 * math.pi ** 2 * Ea * Eb * jac)
                maxres = max(maxres, abs(np.linalg.norm(pb[1:]) ** 2 + p.mb * p.mb - Eb * Eb) / (Eb * Eb))
    contractions = np.real(np.sum(MET[:, None] * MET[None, :] * L[None, :, :] * W, axis=(1, 2)))
    lepton_weights = np.array([expit((p.El - p.mul) / p.T), expit((p.mul - p.El) / p.T)])
    coeff = GF ** 2 * VUD ** 2 / (32 * math.pi ** 2) * pl / p.Enu * contractions * lepton_weights / HBAR
    if not np.all(np.isfinite(coeff)) or np.any(coeff <= 0):
        raise ArithmeticError('Selected ordinary-positive oracle cannot certify underflow/overflow/empty support')
    return dict(chi=float(coeff[0]), j=float(coeff[1]), W=W, contractions=contractions, Emin=E0, umax=umax, max_shell_residual=maxres, min_jacobian=minjac, panels=len(cuts) - 1, order=order, nphi=nphi, tail=tail, q=q, w=w, lepton_tensor=L)
