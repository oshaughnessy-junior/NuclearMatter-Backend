# Generatedselecteddefinitions; environmentexplicitinoracles.py; notupstreamdriver.
def dot(x, y):
    return x[:, 0] * y[:, 0] - np.sum(x[:, 1:] * y[:, 1:], axis=1)

def slash(p, m=0):
    return np.einsum('nu,u,uab->nab', p, E.MET, E.G) + m * E.I

def trace_amp(k, n, l, p):
    V = E.vertices(p - n, C.MW)
    AV = E.adj(V)
    L = E.G @ (E.I - E.G5)
    AL = E.adj(L[None, ...])[0]
    H = np.einsum('nab,nubc,ncd,nvda->nuv', slash(n, C.MN), AV, slash(p, C.MP), V, optimize=True)
    LT = np.einsum('nab,ubc,ncd,vda->nuv', slash(k), L, slash(l, C.ME), AL, optimize=True)
    val = np.einsum('nuv,nuv,u,v->n', LT, H, E.MET, E.MET)
    if max(abs(val.imag)) > 1e-10 * max(abs(val.real)):
        raise ArithmeticError('Trace imaginary residue')
    return E.GF ** 2 * E.VUD ** 2 / 8 * val.real

def integrate(b, order, angular, phi_order, tail, radiation_mus=None):
    start = time.monotonic()
    xp, wp = leggauss(order)
    xx, ww = leggauss(order)
    ci, wi = leggauss(angular)
    co, wo = leggauss(angular)
    ph = 2 * math.pi * np.arange(phi_order) / phi_order
    kpmax = tail * b.T
    pmax = math.sqrt(kpmax * (kpmax + 2 * C.MP))
    rad = (xp + 1) * pmax / 2
    rw = wp * pmax / 2
    emax = max(C.MN - C.MP, C.ME, b.mu_e) + tail * b.T
    mu = b.equilibrium_mu_nue
    scenarios = radiation_mus if radiation_mus is not None else {'empty': None, 'LTE': mu, 'reverse_heldout': mu + 0.23 * b.T}
    acc = {k: dict(F=np.zeros(6), B=np.zeros(6), net=np.zeros(6), entropy=0.0) for k in scenarios}
    density = 0.0
    maxfour = 0.0
    points = 0
    for pp, prw in zip(rad, rw):
        if time.monotonic() - start > 110:
            raise TimeoutError('Independent phase bounded110s')
        ep = math.hypot(pp, C.MP)
        aa = ep * ep - pp * pp * ci * ci
        smax = C.MP ** 2 + C.ME ** 2 + 2 * (ep * emax - pp * math.sqrt(emax * emax - C.ME * C.ME) * ci)
        vmax = (smax - C.MN ** 2) / (2 * C.MP)
        zmax = np.log1p(vmax / b.T)
        z = (xx[:, None] + 1) * zmax / 2
        vv = b.T * np.expm1(z)
        ds = 2 * C.MP * b.T * np.exp(z) * ww[:, None] * zmax / 2
        s = C.MN ** 2 + 2 * C.MP * vv
        D = (s - C.MP ** 2 - C.ME ** 2) / 2
        rt = np.sqrt(D * D - aa * C.ME * C.ME)
        ee = (ep * D + pp * ci * rt) / aa
        jac = (ep + pp * ci * D / rt) / (2 * aa) * ds
        II, JJ, CC, PH = np.meshgrid(np.arange(order), np.arange(angular), co, ph, indexing='ij')
        ii = II.astype(int).ravel()
        jj = JJ.astype(int).ravel()
        en = ee[ii, jj]
        ct = ci[jj]
        cc = CC.ravel()
        phi = PH.ravel()
        pe = np.sqrt(en * en - C.ME * C.ME)
        weights = jac[ii, jj] * wi[jj] * np.tile(np.repeat(wo, phi_order), order * angular) * 2 * math.pi / phi_order
        N = len(en)
        p = np.tile([ep, 0, 0, pp], (N, 1))
        l = np.column_stack([en, pe * np.sqrt(1 - ct * ct), np.zeros(N), pe * ct])
        total = p + l
        ss = dot(total, total)
        root = np.sqrt(ss)
        pf = (ss - C.MN ** 2) / (2 * root)
        beta = total[:, 1:] / total[:, 0, None]
        axis = beta / np.linalg.norm(beta, axis=1)[:, None]
        ey = np.tile([0.0, 1.0, 0.0], (N, 1))
        ex = np.cross(ey, axis)
        direction = cc[:, None] * axis + np.sqrt(1 - cc * cc)[:, None] * (np.cos(phi)[:, None] * ex + np.sin(phi)[:, None] * ey)
        k = E.boost(np.column_stack([pf, pf[:, None] * direction]), beta)
        n = E.boost(np.column_stack([(ss + C.MN ** 2) / (2 * root), -pf[:, None] * direction]), beta)
        amp = trace_amp(k, n, l, p)
        weight = pp * pp * prw / math.pi ** 2 * pe * en / (2 * math.pi ** 2) * weights * amp * pf / (64 * math.pi ** 2 * root * ep * en * C.HBAR * C.HC ** 3)
        if np.any(weight <= 0):
            raise ArithmeticError('Nonpositive trace event')
        fp = expit((b.mu_p - ep) / b.T)
        fe = expit((b.mu_e - en) / b.T)
        fn = expit((b.mu_n - n[:, 0]) / b.T)
        values = np.column_stack([np.ones(N), k[:, 0], en, np.full(N, ep), n[:, 0], ep + en - n[:, 0]])
        density += pp * pp * prw / (math.pi ** 2 * C.HC ** 3) * fp
        maxfour = max(maxfour, float(np.max(abs(p + l - n - k))))
        points += N
        for name, mur in scenarios.items():
            fr = np.zeros(N) if mur is None else expit((mur - k[:, 0]) / b.T)
            F = fp * fe * (1 - fn) * (1 - fr)
            B = (1 - fp) * (1 - fe) * fn * fr
            delta = np.zeros(N) if mur is None else np.log(F) - np.log(B)
            net = weight * (F - B)
            a = acc[name]
            a['F'] += weight * F @ values
            a['B'] += weight * B @ values
            a['net'] += net @ values
            if mur is not None:
                a['entropy'] += float(net @ delta)
    out = {k: {a: v.tolist() if isinstance(v, np.ndarray) else v for a, v in z.items()} for k, z in acc.items()}
    return dict(scenarios=out, density=density, maxfour=maxfour, points=points, controls=[order, angular, phi_order, tail], seconds=time.monotonic() - start)
