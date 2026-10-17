# ASTselected puredefinitions, injectedglobalsinprimitives.py; no upstreamdriver
def dot(a, b):
    return a[:, 0] * b[:, 0] - np.sum(a[:, 1:] * b[:, 1:], axis=1)

def scalar_amplitude(k, n, l, p):
    """Guo47 invariants for reciprocal ν+n capture, avg4 for original p+e-.

    Adapted from accepted review_pc_moments polynomial with physical n->p
    ordering and NU (not anti) VA/AF signs. Checked against literal spinors.
 """
    q = p - n
    t = dot(q, q)
    if np.any(t >= min(139.57 ** 2, 840.0 ** 2, 1000.0 ** 2, 4 * MW * MW)):
        raise ValueError('Subpole free-current events required')
    den = (1 - t / (4 * MW * MW)) * (1 - t / 840 ** 2) ** 2
    v = (1 - 4.706 * t / (4 * MW * MW)) / den
    a = 1.2723 / (1 - t / 1000 ** 2) ** 2
    f = 3.706 / den
    g = 2 * MW * MW * a / (139.57 ** 2 - t)
    kn, kp, ln, lp, kl, np_ = [dot(x, y) for x, y in [(k, n), (k, p), (l, n), (l, p), (k, l), (n, p)]]
    kq, lq, nq, pq = [dot(x, q) for x in [k, l, n, p]]
    ma = MN
    mb = MP
    VV = v * v * (kn * lp + kp * ln - ma * mb * kl)
    VA = 2 * v * a * (kn * lp - kp * ln)
    AA = a * a * (kn * lp + kp * ln + ma * mb * kl)
    VF = v * f / (2 * MW) * ((kn * mb - kp * ma) * lq + (ln * mb - lp * ma) * kq + (nq * mb - pq * ma) * kl)
    AF = a * f / MW * ((kn * mb + kp * ma) * lq - (ln * mb + lp * ma) * kq)
    FF = f * f / (8 * MW * MW) * (2 * (ln * pq + lp * nq) * kq + 2 * (kn * pq + kp * nq) * lq - 2 * kq * lq * np_ + t * (kl * np_ - 2 * kn * lp - 2 * kp * ln) - ma * mb * (kl * t + 2 * kq * lq))
    AP = a * g / MW * (kq * (ln * mb - lp * ma) + lq * (kn * mb - kp * ma) + kl * (pq * ma - nq * mb))
    PP = g * g / (2 * MW * MW) * (2 * kq * lq - t * kl) * (np_ - ma * mb)
    return 8 * E.GF ** 2 * E.VUD ** 2 * (VV + VA + AA + VF + AF + FF + AP + PP)
