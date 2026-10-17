# Generatedselecteddefinitions; environmentexplicitinoracles.py; notupstreamdriver.
def events(order, orient, phi):
    x, w = leggauss(order)
    Emax = ((MN - MP) * (MN + MP) + ME * ME) / (2 * MN)
    lim = math.sqrt(Emax - ME)
    z = (x + 1) * lim / 2
    Ee = ME + z * z
    we = w * lim * z
    E, X = np.meshgrid(Ee, x, indexing='ij')
    e = E.ravel()
    pe = np.sqrt((e - ME) * (e + ME))
    N = MN * (Emax - e)
    low = N / (MN - e + pe)
    high = N / (MN - e - pe)
    enu = (low + high) / 2 + (high - low) / 2 * X.ravel()
    cos = (N / enu - (MN - e)) / pe
    if np.any(abs(cos) > 1 + 1e-08):
        raise ArithmeticError('Dalitz event off physical cone')
    cos = np.clip(cos, -1, 1)
    weights = (we[:, None] * w[None, :]).ravel() * (high - low) / 2
    pn = np.tile([MN, 0, 0, 0], (len(e), 1))
    l = np.column_stack([e, np.zeros(len(e)), np.zeros(len(e)), pe])
    k = np.column_stack([enu, enu * np.sqrt(1 - cos * cos), np.zeros(len(e)), enu * cos])
    pp = pn - l - k
    ch = C.V.Channel(MN, MP, ME)
    amps = []
    for i in range(len(e)):
        p = dict(k=k[i], pa=pn[i], l=l[i], pb=pp[i])
        q = pp[i] - pn[i]
        t = q[0] ** 2 - q[1:] @ q[1:]
        ff = C.FormFactors().values(t, (MN + MP) / 2)
        amps.append(sum(C.guo_invariants(ch, p, ff).values()))
    restweight = weights * np.array(amps) / (64 * math.pi ** 3 * MN * HBAR)
    if np.any(restweight <= 0):
        raise ArithmeticError('Positive source-invariant event weight unresolved')
    u, wu = leggauss(orient)
    ph = 2 * math.pi * np.arange(phi) / phi
    U, PH = np.meshgrid(u, ph, indexing='ij')
    u = U.ravel()
    ph = PH.ravel()
    sinu = np.sqrt(1 - u * u)
    ow = np.repeat(wu / (2 * phi), phi)
    rest = {}
    electron = np.stack([np.broadcast_to(e[:, None], (len(e), len(u))), pe[:, None] * sinu[None, :], np.zeros((len(e), len(u))), pe[:, None] * u[None, :]], axis=-1)
    sc = np.sqrt(1 - cos * cos)[:, None]
    cv = cos[:, None]
    Kx = enu[:, None] * (cv * sinu[None, :] + sc * np.cos(ph)[None, :] * u[None, :])
    Ky = enu[:, None] * sc * np.sin(ph)[None, :]
    Kz = enu[:, None] * (cv * u[None, :] - sc * np.cos(ph)[None, :] * sinu[None, :])
    nu = np.stack([np.broadcast_to(enu[:, None], Kx.shape), Kx, Ky, Kz], axis=-1)
    neutron = np.broadcast_to(np.array([MN, 0, 0, 0]), electron.shape)
    proton = neutron - electron - nu
    for name, v in [('pn', neutron), ('pp', proton), ('e', electron), ('nu', nu)]:
        rest[name] = v.reshape((-1, 4)).copy()
    rest['weights'] = (restweight[:, None] * ow[None, :]).ravel()
    rest['proper_rate'] = float(sum(restweight))
    rest['proper_moments'] = dict(neutrino_energy=float(restweight @ enu), electron_energy=float(restweight @ e), proton_energy=float(restweight @ pp[:, 0]))
    return rest

def integrate(bath, muanti, order=24, radial=12, orient=10, phi=12, checkpoint=None, budget=105, radiation_mus=None, parent_tail=80.0):
    start = time.monotonic()
    rest = events(order, orient, phi)
    x, w = leggauss(radial)
    out = {name: dict(F=np.zeros(6), B=np.zeros(6), net=np.zeros(6), entropy=0.0, max_log_ratio_residual=0.0) for name in ['unblocked'] + list(radiation_mus)}
    density = 0.0
    properdensity = 0.0
    minoccup = 1.0
    maxshell = 0.0
    edges = [0.0, 0.7, 1.7, 3.0, 5.0, math.sqrt(parent_tail)]
    T = bath.T
    for panel, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        for y, rw in zip((lo + hi) / 2 + (hi - lo) * x / 2, w * (hi - lo) / 2):
            if time.monotonic() - start > budget:
                raise TimeoutError('Independent moving-event budget')
            p = math.sqrt(2 * MN * T) * y
            En = math.hypot(MN, p)
            gamma = En / MN
            vel = p / En
            parent = p * p * math.sqrt(2 * MN * T) * rw / (math.pi ** 2 * HC ** 3)
            fn = float(log_expit((bath.mu_n - En) / T))
            nfn = float(log_expit((En - bath.mu_n) / T))
            density += parent * math.exp(fn)
            properdensity += parent * math.exp(fn) / gamma
            lab = {name: np.column_stack([gamma * (v[:, 0] + vel * v[:, 3]), v[:, 1], v[:, 2], gamma * (v[:, 3] + vel * v[:, 0])]) for name, v in rest.items() if isinstance(v, np.ndarray) and v.ndim == 2}
            maxshell = max(maxshell, float(np.max(abs(lab['pn'] - lab['pp'] - lab['e'] - lab['nu']))))
            Ep, Ee, Ev = (lab[n][:, 0] for n in ['pp', 'e', 'nu'])
            Enarray = lab['pn'][:, 0]
            direct = Enarray - Ep - Ee
            E = np.column_stack([np.ones(len(Ev)), Ev, Ee, Ep, Enarray, direct])
            logbase = np.log(rest['weights']) + math.log(parent / gamma)
            fp = log_expit((bath.mu_p - Ep) / T)
            nfp = log_expit((Ep - bath.mu_p) / T)
            fe = log_expit((bath.mu_e - Ee) / T)
            nfe = log_expit((Ee - bath.mu_e) / T)
            for name in out:
                if name == 'unblocked':
                    lf = fn + np.zeros(len(Ev))
                    lb = None
                elif name == 'empty':
                    lf = fn + nfp + nfe
                    lb = None
                else:
                    chemical = radiation_mus[name]
                    fv = log_expit((chemical - Ev) / T)
                    nfv = log_expit((Ev - chemical) / T)
                    lf = fn + nfp + nfe + nfv
                    lb = nfn + fp + fe + fv
                    force = (bath.mu_n - bath.mu_p - bath.mu_e - chemical) / T
                    delta = lf - lb
                    out[name]['max_log_ratio_residual'] = max(out[name]['max_log_ratio_residual'], float(np.max(abs(delta - force))))
                forward = np.exp(logbase + lf)
                out[name]['F'] += forward @ E
                if lb is not None:
                    backward = np.exp(logbase + lb)
                    out[name]['B'] += backward @ E
                    net = np.sign(delta) * np.exp(logbase + np.maximum(lf, lb)) * -np.expm1(-abs(delta))
                    out[name]['entropy'] += float(net @ delta)
                else:
                    net = forward
                out[name]['net'] += net @ E
        if checkpoint:
            checkpoint(dict(panel=panel, panels=len(edges) - 1, partial=out, density=density))
    return dict(scenarios=out, names=NAMES, density=density, proper_density=properdensity, rest_rate=rest['proper_rate'], rest_moments=rest['proper_moments'], max_fourmomentum_residual=maxshell, controls=dict(rest=order, parent_per_panel=radial, orient=orient, phi=phi, parent_p_over_sqrt2MT_max=edges[-1]), seconds=time.monotonic() - start)
