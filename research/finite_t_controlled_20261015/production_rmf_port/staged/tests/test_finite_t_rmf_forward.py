"""Independent original-momentum/grand-potential forward oracle, no candidate imports."""
from pathlib import Path
import json, math
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import expit
import pytest
HC = 197.3269804
M = 939./HC
CS, CO, CR, B, C = 11.785, 7.148, 4.410, .002947, -.001070
POINTS = [(10., .16, .1), (20., .24, .3), (14., .19, .22),
          (7., .18, .17), (25., .28, .35)]


def oracle(T, nB, Y, *, tail=85., tol=3e-12):
    """Nested scalar density roots, direct p quadrature and pressure from logZ."""
    t = T/HC
    def integrate(nu, m, kind):
        end = max(abs(nu), m)+tail*t
        pmax = math.sqrt((end-m)*(end+m))
        splits = [0., pmax]
        for e in [abs(nu)-10*t, abs(nu), abs(nu)+10*t]:
            if m < e < end:
                splits.append(math.sqrt((e-m)*(e+m)))
        splits = sorted(set(splits))
        def integrand(p):
            e = math.hypot(p, m)
            x, y = (e-nu)/t, (e+nu)/t
            a, b = expit(-x), expit(-y)
            if kind == 'n':value = a-b
            elif kind == 'scalar':value = m/e*(a+b)
            elif kind == 'energy':value = e*(a+b)
            elif kind == 'pressure':value = t*(np.logaddexp(0., -x)+np.logaddexp(0., -y))
            elif kind == 'entropy':
                value = sum(np.logaddexp(0., -abs(z))+abs(z)*expit(-abs(z)) for z in [x,y])
            else:raise ValueError(kind)
            return p*p*value/math.pi**2
        return sum(quad(integrand, lo, hi, epsabs=2e-15, epsrel=tol, limit=250)[0]
                   for lo, hi in zip(splits[:-1], splits[1:]))
    def species(d, m):
        nu = brentq(lambda v:integrate(v,m,'n')/d-1, 0., 2*M, xtol=3e-13, rtol=2e-14)
        return nu
    def scalar(phi):
        m=M-phi
        return phi/CS+B*M*phi**2+C*phi**3-sum(integrate(species(d,m),m,'scalar')for d in [nB*(1-Y),nB*Y])
    grid = np.linspace(0., .985*M, 73)
    vals = [scalar(ph)for ph in grid]
    brackets = [(a,b)for a,b,fa,fb in zip(grid[:-1],grid[1:],vals[:-1],vals[1:])if fa<0<=fb]
    if len(brackets)!=1:raise ArithmeticError('Independent unique scalar bracket absent')
    ph=brentq(scalar,*brackets[0],xtol=3e-13,rtol=2e-14)
    m=M-ph
    ns=[species(d,m)for d in [nB*(1-Y),nB*Y]]
    gas=[{k:integrate(v,m,k)for k in ['n','scalar','energy','pressure','entropy']}for v in ns]
    us=ph**2/(2*CS)+B*M*ph**3/3+C*ph**4/4
    uv=CO*nB*nB/2+CR*((2*Y-1)*nB)**2/8
    W,R=CO*nB,CR*(2*Y-1)*nB/2
    energy=(sum(g['energy']for g in gas)+us+uv)*HC
    s=sum(g['entropy']for g in gas)
    return dict(T=T,nB=nB,Yp=Y,phi=ph*HC,dirac_mass=m*HC,
        mu_n=(ns[0]+W-R/2)*HC,mu_p=(ns[1]+W+R/2)*HC,
        kinetic_mu_n=ns[0]*HC,kinetic_mu_p=ns[1]*HC,
        energy_density=energy,pressure=(sum(g['pressure']for g in gas)-us+uv)*HC,
        entropy_density=s,free_energy_density=energy-T*s,
        neutron_density=gas[0]['n'],proton_density=gas[1]['n'],scalar_density=sum(g['scalar']for g in gas),
        bracket=list(map(float,brackets[0])),root_residual=float(scalar(ph))/nB)


@pytest.mark.parametrize('T,n,Y',POINTS)
def test_forward_original_momentum(T,n,Y):
    expected=oracle(T,n,Y)
    from nmbackend.reference.finite_t_rmf import state_density
    s=state_density(T,n,Y)
    for k in ['phi','dirac_mass','mu_n','mu_p','kinetic_mu_n','kinetic_mu_p',
              'energy_density','pressure','entropy_density','free_energy_density',
              'neutron_density','proton_density','scalar_density']:
        np.testing.assert_allclose(getattr(s,k),expected[k],rtol=2e-9,atol=1e-10)
    assert s.locally_admitted
    assert s.normalized_equation_residual<2e-9


def test_saved_reference_regression():
    from nmbackend.reference.finite_t_rmf import state_density
    p=Path(__file__).parent/'data/finite_t_gm1_reference.json'
    for row in json.loads(p.read_text())['rows']:
        z=row['state'];s=state_density(z['temperature'],z['nB'],z['Yp'])
        for new,old in [('phi','phi_mev'),('dirac_mass','dirac_mass_mev'),('mu_n','mu_n_mev'),('mu_p','mu_p_mev'),('energy_density','energy_density'),('pressure','pressure'),('entropy_density','entropy_density')]:
            np.testing.assert_allclose(getattr(s,new),z[old],rtol=2e-9,atol=1e-10)
