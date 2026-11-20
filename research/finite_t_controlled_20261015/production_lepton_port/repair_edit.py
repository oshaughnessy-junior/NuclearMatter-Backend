from pathlib import Path
p=Path('outputs/nmb-finite-t-20261015/production_lepton_port/staged/src/nmbackend/reference/finite_t_leptons.py')
s=p.read_text().replace('from scipy.special import expit','from scipy.special import expit, log_expit')
a=s.index('def state_mu('); b=s.index('\ndef state_density(',a)
s=s[:a]+'''def _restore(value, log_scale):
    # Combine dimensional and occupation scales before rounding to float64.
    # A subnormal/zero output has no relative-precision guarantee.
    if value == 0:
        return 0.0
    try:
        return math.copysign(math.exp(math.log(abs(value))+log_scale), value)
    except OverflowError as exc:
        raise ArithmeticError("Unrepresented finite-temperature moment") from exc


def _occupation_ratio(x, edge):
    """f(edge-x)/f(edge), x>=0, without a dilute exponential floor."""
    if edge <= 0:
        return math.exp(-x)*(1+math.exp(edge))/(1+math.exp(edge-x))
    return expit(edge-x)/expit(edge)


def _panels(end, edge):
    return sorted(set([0.0, end]+[x for x in
                       (edge-30.0, edge, edge+30.0) if 0 < x < end]))


def state_mu(temperature_mev, chemical_potential_mev, mass_mev=M_E, *,
             rtol=2e-12, tail=60.0):
    """Evaluate one ideal pair gas at T>0 and total mu (both MeV).

    Signed mu is allowed; mu=0 retains a net-zero pair plasma. Photons,
    interactions and weak equilibrium are absent. Ordinary populations may
    underflow; no relative accuracy or exact absence is implied by such a zero.
    Unresolved susceptibility or nonfinite responses raise ArithmeticError.
    """
    T, mu, m, rtol, tail = _inputs(temperature_mev, chemical_potential_mev,
                                 mass_mev, rtol, tail)
    sign = 1.0 if mu >= 0 else -1.0
    v, u = m/T, abs(mu)/T
    edge = (abs(mu)-m)/T
    end = max(0.0, edge)+tail
    if not all(math.isfinite(x) for x in (v, u, edge, end)):
        raise ArithmeticError("Unrepresented finite-temperature energy interval")
    log_a, log_b = float(log_expit(edge)), float(log_expit(-u-v))
    log_pref = 3*math.log(T)+math.log(_MEASURE)
    energy_scale = max(1.0, v, u)
    response_scale = max(1.0, -edge)
    breaks = _panels(end, edge)

    def geometry(x):
        energy = v+x
        momentum = math.sqrt(x*(x+2*v))
        return energy, momentum, momentum*energy

    def bulk(pop_edge):
        def integrand(x):
            energy, momentum, measure = geometry(x)
            occupation = _occupation_ratio(x, pop_edge)
            return measure*occupation*np.array([
                1.0, energy/energy_scale,
                momentum*momentum/(3*energy*energy_scale) if energy else 0.0])
        # Zero absolute tolerance: each population is dimensionless and scaled
        # separately, including a dilute antiparticle beside a degenerate gas.
        return _integrate(integrand, breaks, rtol, 0.0)

    particle, anti = bulk(edge), bulk(-u-v)
    npart = _restore(particle[0], log_pref+log_a)
    nanti = _restore(anti[0], log_pref+log_b)
    energy = (_restore(particle[1], log_pref+log_a+math.log(T)+math.log(energy_scale))
              + _restore(anti[1], log_pref+log_b+math.log(T)+math.log(energy_scale)))
    pressure = (_restore(particle[2], log_pref+log_a+math.log(T)+math.log(energy_scale))
                + _restore(anti[2], log_pref+log_b+math.log(T)+math.log(energy_scale)))
    ratio_b = math.exp(log_b-log_a)
    net_factor = -math.expm1(-2*u)

    def entropy_scaled(z, log_population):
        q = abs(z)
        e = math.exp(-q)
        quotient = math.log1p(e)/e if e else 1.0
        return math.exp(-q-log_population)*(quotient+q/(1+e))

    def response(x):
        _, _, measure = geometry(x)
        a = _occupation_ratio(x, edge)
        b = ratio_b*_occupation_ratio(x, -u-v)
        d = x-edge
        wa, wb = a*expit(d), b*expit(x+u+v)
        net = a/(1+math.exp(-(x+u+v)))
        if u < 0.1:
            # w(a)-w(b)=(a-b)(1-a-b); retain tiny signed mu.
            difference_w = net*net_factor*(1-expit(-d)-expit(-(x+u+v)))
            first = d*difference_w-2*u*wb
        else:
            first = d*wa-(d+2*u)*wb
        second = d*d*wa+(d+2*u)**2*wb
        entropy = (entropy_scaled(d, log_a)
                   + ratio_b*entropy_scaled(x+u+v, log_b))
        return measure*np.array([net, wa+wb, first/response_scale,
                                 second/response_scale**2, entropy/response_scale])

    net, a0, a1, a2, entropy = _integrate(response, breaks, rtol, 0.0)
    chi = _restore(a0, log_pref+log_a-math.log(T))
    if a0 <= 0 or chi <= 0:
        raise ArithmeticError("Susceptibility underflow; state outside reference domain")
    covariance = a2-a1*a1/a0
    if covariance < -1e-12*max(a2, 1e-300):
        raise ArithmeticError("Unresolved negative fixed-net-number heat capacity")
    n = _restore(net*net_factor, log_pref+log_a)
    entropy = _restore(entropy, log_pref+log_a+math.log(response_scale))
    cv = _restore(max(0.0, covariance),
                  log_pref+log_a+2*math.log(response_scale))
    n_temperature = _restore(a1, log_pref+log_a+math.log(response_scale)-math.log(T))
    result = LeptonState(T, mu, m, sign*n,
                         npart if sign > 0 else nanti,
                         nanti if sign > 0 else npart,
                         energy, pressure, entropy, energy-T*entropy,
                         cv, chi, sign*n_temperature,
                         -sign*response_scale*a1/a0)
    if not all(math.isfinite(float(x)) for x in result.asdict().values()):
        raise ArithmeticError("Nonfinite finite-temperature lepton response")
    return result

''' + s[b:]
a=s.index('    u = abs(state.chemical_potential)',s.index('def thermal_energy_density'))
s=s[:a]+'''    u, v = abs(state.chemical_potential)/T, m/T
    edge = (abs(state.chemical_potential)-m)/T
    pf = HBARC*(3*math.pi**2*abs(state.net_density))**(1.0/3.0)/T
    cold_edge = pf*pf/(math.hypot(pf, v)+v) if pf else 0.0
    end = max(0.0, edge, cold_edge)+tail
    if not all(math.isfinite(x) for x in (u, v, cold_edge, end)):
        raise ArithmeticError("Unrepresented thermal energy interval")
    log_pref = 4*math.log(T)+math.log(_MEASURE)

    def measure(x):
        return math.sqrt(x*(x+2*v))*(v+x)

    # The fixed-number subtraction is a sum of positive holes below the cold
    # surface, particle excitations above it, and antiparticle energy. Each
    # occupation is scaled at its own integration-domain maximum.
    holes = 0.0
    if cold_edge > 0:
        log_h = float(log_expit(cold_edge-edge))
        def hole_integrand(x):
            vacancy = math.exp(float(log_expit(x-edge))-log_h)
            return measure(x)*(cold_edge-x)*vacancy
        breaks = sorted(set([0.0, cold_edge]+[x for x in
                            (edge-30.0, edge, edge+30.0) if 0 < x < cold_edge]))
        holes = _restore(float(_integrate(hole_integrand, breaks, rtol, 0.0)),
                         log_pref+log_h)
    log_exc = float(log_expit(edge-cold_edge))
    def excitation(y):
        return measure(y+cold_edge)*y*_occupation_ratio(y, edge-cold_edge)
    excitation_end = end-cold_edge
    excited = _restore(float(_integrate(excitation,
                       _panels(excitation_end, edge-cold_edge), rtol, 0.0)),
                       log_pref+log_exc)
    log_anti = float(log_expit(-u-v))
    def antiparticle(x):
        return measure(x)*(x+cold_edge+2*v)*_occupation_ratio(x, -u-v)
    anti = _restore(float(_integrate(antiparticle, _panels(end, edge), rtol, 0.0)),
                    log_pref+log_anti)
    return holes+excited+anti
'''
p.write_text(s)
