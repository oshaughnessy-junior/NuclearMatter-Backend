# ASTselected puredefinitions, injectedglobalsinprimitives.py; no upstreamdriver
def dot(a, b):
    return a[:, 0] * b[:, 0] - np.sum(a[:, 1:] * b[:, 1:], axis=1)

def amplitude(k, n, l, b):
    q = b - n
    t = dot(q, q)
    den = (1 - t / (4 * MW * MW)) * (1 - t / 840 ** 2) ** 2
    v = (1 - 4.706 * t / (4 * MW * MW)) / den
    a = 1.2723 / (1 - t / 1000 ** 2) ** 2
    f = 3.706 / den
    g = 2 * MW * MW * a / (139.57 ** 2 - t)
    kn, kb, ln, lb, kl, nb = [dot(x, y) for x, y in [(k, n), (k, b), (l, n), (l, b), (k, l), (n, b)]]
    kq, lq, nq, bq = [dot(x, q) for x in [k, l, n, b]]
    ma = MP
    mb = MN
    pref = 16 * GF ** 2 * VUD ** 2
    VV = v * v * (kn * lb + kb * ln - ma * mb * kl)
    VA = -2 * v * a * (kn * lb - kb * ln)
    AA = a * a * (kn * lb + kb * ln + ma * mb * kl)
    VF = v * f / (2 * MW) * ((kn * mb - kb * ma) * lq + (ln * mb - lb * ma) * kq + (nq * mb - bq * ma) * kl)
    AF = -a * f / MW * ((kn * mb + kb * ma) * lq - (ln * mb + lb * ma) * kq)
    FF = f * f / (8 * MW * MW) * (2 * (ln * bq + lb * nq) * kq + 2 * (kn * bq + kb * nq) * lq - 2 * kq * lq * nb + t * (kl * nb - 2 * kn * lb - 2 * kb * ln) - ma * mb * (kl * t + 2 * kq * lq))
    AP = a * g / MW * (kq * (ln * mb - lb * ma) + lq * (kn * mb - kb * ma) + kl * (bq * ma - nq * mb))
    PP = g * g / (2 * MW * MW) * (2 * kq * lq - t * kl) * (nb - ma * mb)
    return pref * (VV + VA + AA + VF + AF + FF + AP + PP) / 2
