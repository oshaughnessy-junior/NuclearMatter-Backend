"""Owned configured currents; generated from pinned source functions, not a production kernel."""
import numpy as np
MN=939.56542052;MP=938.27208816;ME=.51099895069;MW=(MN+MP)/2;GF=1.1663785e-11;VUD=.974
def dot(a,b):return a[:,0]*b[:,0]-np.sum(a[:,1:]*b[:,1:],axis=1)
def ec_full(k, n, l, p):
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
    return 8 * GF ** 2 * VUD ** 2 * (VV + VA + AA + VF + AF + FF + AP + PP)

def ec_F2zero(k, n, l, p):
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
    f = 0.0
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
    return 8 * GF ** 2 * VUD ** 2 * (VV + VA + AA + VF + AF + FF + AP + PP)

def pc_full(k, n, l, b):
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

def pc_F2zero(k, n, l, b):
    q = b - n
    t = dot(q, q)
    den = (1 - t / (4 * MW * MW)) * (1 - t / 840 ** 2) ** 2
    v = (1 - 4.706 * t / (4 * MW * MW)) / den
    a = 1.2723 / (1 - t / 1000 ** 2) ** 2
    f = 0.0
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
