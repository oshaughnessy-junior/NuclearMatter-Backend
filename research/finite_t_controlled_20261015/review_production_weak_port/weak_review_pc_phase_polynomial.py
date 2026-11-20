# Generatedselecteddefinitions; environmentexplicitinoracles.py; notupstreamdriver.
def dot(a, b):
    return a[:, 0] * b[:, 0] - np.sum(a[:, 1:] * b[:, 1:], axis=1)

def boost(p, b):
    b2 = np.sum(b * b, axis=1)
    g = 1 / np.sqrt(1 - b2)
    bp = np.sum(b * p[:, 1:], axis=1)
    return np.column_stack([g * (p[:, 0] + bp), p[:, 1:] + (g * g / (g + 1) * bp + g * p[:, 0])[:, None] * b])

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

def integral(b, order_n=24, order_e=36, order_angle=10, tail=50.0, budget=105.0, radiation_mus=None):
    start = time.monotonic()
    T = b['T']
    mun, mup, mue = [b[x] for x in ['mu_n', 'mu_p', 'mu_e']]
    mu_eq = mun - mup - mue
    scenarios = radiation_mus if radiation_mus is not None else [None, mu_eq, mu_eq - 0.4 * T]
    x, w = leggauss(order_n)
    y, v = leggauss(order_e)
    c, cw = leggauss(order_angle)
    ph = 2 * math.pi * np.arange(order_angle) / order_angle
    Kn = max(0.0, mun - MN, mup - MP) + tail * T + max(0.0, mu_eq)
    pnmax = math.sqrt(Kn * (Kn + 2 * MN))
    pscale = math.sqrt(2 * MN * T)
    anglemax = math.atan(pnmax / pscale)
    a = (x + 1) * anglemax / 2
    pnodes = pscale * np.tan(a)
    nw = w * anglemax / 2 * pscale / np.cos(a) ** 2
    Ke = max(0.0, -mue - ME, mu_eq - (MN - MP) - ME) + tail * T
    pemax = math.sqrt(Ke * (Ke + 2 * ME))
    emax = math.log1p(pemax / T)
    ee = (y + 1) * emax / 2
    ep = T * np.expm1(ee)
    ew = v * emax / 2 * T * np.exp(ee)
    PE, CI, CO, PH = np.meshgrid(ep, c, c, ph, indexing='ij')
    pe = PE.ravel()
    ci = CI.ravel()
    co = CO.ravel()
    phi = PH.ravel()
    El = np.hypot(pe, ME)
    N = len(El)
    weights = (ew[:, None, None, None] * cw[None, :, None, None] * cw[None, None, :, None] * np.ones((1, 1, 1, order_angle)) * 2 * math.pi / order_angle).ravel()
    sums = np.zeros((len(scenarios), 2, 6))
    direct = np.zeros((len(scenarios), 6))
    entropy = np.zeros(len(scenarios))
    dens = 0.0
    shell = 0.0
    ampmin = math.inf
    for p, wp in zip(pnodes, nw):
        if time.monotonic() - start > budget:
            raise TimeoutError('Independent total quadrature budget105s; no completion')
        En = math.hypot(p, MN)
        nn = np.tile([En, 0, 0, p], (N, 1))
        l = np.column_stack([El, pe * np.sqrt(1 - ci * ci), np.zeros(N), pe * ci])
        P = nn + l
        s = dot(P, P)
        rt = np.sqrt(s)
        beta = P[:, 1:] / P[:, 0, None]
        lcm = boost(l, -beta)
        axis = lcm[:, 1:] / np.linalg.norm(lcm[:, 1:], axis=1)[:, None]
        ey = np.tile([0.0, 1.0, 0.0], (N, 1))
        ex = np.cross(ey, axis)
        direc = co[:, None] * axis + np.sqrt(1 - co * co)[:, None] * (np.cos(phi)[:, None] * ex + np.sin(phi)[:, None] * ey)
        pf = (s - MP * MP) / (2 * rt)
        k = boost(np.column_stack([pf, pf[:, None] * direc]), beta)
        pb = boost(np.column_stack([(s + MP * MP) / (2 * rt), -pf[:, None] * direc]), beta)
        amp = amplitude(k, pb, l, nn)
        ampmin = min(ampmin, float(np.min(amp)))
        weight = p * p * wp / math.pi ** 2 * pe * pe / (2 * math.pi ** 2) * weights * amp * pf / (64 * math.pi ** 2 * rt * En * El * HBAR * HC ** 3)
        fn = expit((mun - En) / T)
        fp = expit((mup - pb[:, 0]) / T)
        fl = expit((-mue - El) / T)
        nv = expit((En - mun) / T)
        pv = expit((pb[:, 0] - mup) / T)
        lv = expit((El + mue) / T)
        dens += p * p * wp / math.pi ** 2 / HC ** 3 * fn
        values = np.column_stack([np.ones(N), k[:, 0], El, pb[:, 0], np.full(N, En), En + El - pb[:, 0]])
        for z, mu in enumerate(scenarios):
            fr = np.zeros(N) if mu is None else expit((mu - k[:, 0]) / T)
            vr = np.ones(N) if mu is None else expit((k[:, 0] - mu) / T)
            F = fn * fl * pv * vr
            B = nv * lv * fp * fr
            WF = weight * F
            WB = weight * B
            sums[z, 0] += WF @ values
            sums[z, 1] += WB @ values
            direct[z] += (WF - WB) @ values
            if mu is not None:
                entropy[z] += float(np.sum((WF - WB) * (np.log(F) - np.log(B))))
        shell = max(shell, float(np.max(abs(nn + l - pb - k))) / MN, float(np.max(abs(dot(pb, pb) - MP * MP))) / (MP * MP))
    return dict(sums=sums.tolist(), net=direct.tolist(), entropy=entropy.tolist(), neutron_density=dens, max_shell=shell, min_amplitude=ampmin, controls=dict(n=order_n, e=order_e, angle=order_angle, tail=tail, maps='neutronp tangent,positronp logarithm,outangle incidentpositronCMaxis'), seconds=time.monotonic() - start, units='number fm^-3s^-1,energy MeVfm^-3s^-1')
