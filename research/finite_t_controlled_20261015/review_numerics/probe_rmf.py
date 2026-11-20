"""Independent finite-T RMF grand-potential and response checks."""
from pathlib import Path
import sys,math,json,hashlib
import numpy as np
from scipy.integrate import quad

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'rmf_reference'))
import gm1_finite_t as r
checks=[]; samples=[]
def check(name,actual,expected,tol=1e-5,scale=1e-20):
    err=abs(actual-expected)/max(abs(expected),scale)
    checks.append(dict(name=name,actual=float(actual),expected=float(expected),error=float(err),passed=bool(err<tol)))

def grand_pressure(t,nu,m):
    # Scalar momentum-space grand potential, no energy/entropy-moment reuse.
    endE=max(m,abs(nu))+90*t
    end=math.sqrt(endE**2-m*m)
    points=[math.sqrt(e*e-m*m) for e in [abs(nu)-10*t,abs(nu),abs(nu)+10*t] if m<e<endE]
    f=lambda p:p*p*(np.logaddexp(0,(nu-math.hypot(p,m))/t)+np.logaddexp(0,(-nu-math.hypot(p,m))/t))
    return t/math.pi**2*quad(f,0,end,points=points,epsabs=1e-15,epsrel=1e-11,limit=300)[0]

# Verify pinned snapshot identity using Git's actual blob hash convention.
for name,expected in [('rmft.py','1edba14f7422f37f1ae6e4eb58167dc2c2140061'),('constants.py','cbd0b9009e1f333c6c447930670bcbfedaefd68e')]:
    data=(ROOT/'rmf_reference/source_snapshot'/name).read_bytes()
    actual=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
    checks.append(dict(name='pinned Git blob '+name,actual=actual,expected=expected,passed=actual==expected))

for n,T,Y in [(.12,15.,.3),(.4,50.,.2),(.02,3.,.2)]:
    s=r.state(n,T,Y); samples.append(s.asdict())
    m=s.dirac_mass_mev/r.HBARC; tf=T/r.HBARC
    nuclear_pressure=sum(grand_pressure(tf,nu/r.HBARC,m) for nu in [s.kinetic_mu_n_mev,s.kinetic_mu_p_mev])*r.HBARC-s.scalar_field_energy_density+s.vector_field_energy_density
    check(f'grand-potential pressure {n,T,Y}',s.pressure,nuclear_pressure,3e-9,1e-7)
    hn=n*8e-5; ht=T*8e-4; hy=8e-5
    a=r.state(n-hn,T,Y,initial=s); b=r.state(n+hn,T,Y,initial=s)
    c=r.state(n,T-ht,Y,initial=s); d=r.state(n,T+ht,Y,initial=s)
    e=r.state(n,T,Y-hy,initial=s); f=r.state(n,T,Y+hy,initial=s)
    check(f'pressure Helmholtz derivative {n,T,Y}',n*(b.free_energy_density-a.free_energy_density)/(2*hn)-s.free_energy_density,s.pressure,1e-5,1e-6)
    check(f'composition entropy Maxwell {n,T,Y}',(f.entropy_density-e.entropy_density)/(2*hy),-n*((d.mu_p_mev-d.mu_n_mev)-(c.mu_p_mev-c.mu_n_mev))/(2*ht),3e-5,1e-6)
    # At fixed species density, derivative of pressure obeys Gibbs-Duhem.
    dnmun=(b.mu_n_mev-a.mu_n_mev)/(2*hn); dnmup=(b.mu_p_mev-a.mu_p_mev)/(2*hn)
    check(f'Gibbs-Duhem density pressure {n,T,Y}',(b.pressure-a.pressure)/(2*hn),n*((1-Y)*dnmun+Y*dnmup),3e-5,1e-7)
    refined=r.state(n,T,Y,rtol=2e-12,tail=80.,initial=s)
    check(f'pressure refinement {n,T,Y}',s.pressure,refined.pressure,1e-8)

# Thermal-antibaryon scalar sign and no-sea convention, including pair symmetry.
for T,nu,m in [(100./r.HBARC,0.,939./r.HBARC),(100./r.HBARC,.5,300./r.HBARC)]:
    g=r.gas(T,nu,m)
    h=1e-4*T
    check(f'gas n=dP/dnu T{T} nu{nu}',g['n'],(grand_pressure(T,nu+h,m)-grand_pressure(T,nu-h,m))/(2*h),2e-6,1e-12)
    check(f'gas scalar=-dP/dmass T{T} nu{nu}',g['scalar'],-(grand_pressure(T,nu,m+h)-grand_pressure(T,nu,m-h))/(2*h),2e-6)
    check(f'gas s=dP/dT T{T} nu{nu}',g['entropy'],(grand_pressure(T+h,nu,m)-grand_pressure(T-h,nu,m))/(2*h),2e-6)
    check(f'gas Euler T{T} nu{nu}',g['energy']+g['pressure'],T*g['entropy']+nu*g['n'],1e-10)
    if nu==0:
        check('pair symmetry net zero',g['n'],0.,1e-10,1e-20)
        checks.append(dict(name='thermal antiparticles retained',actual=g['antiparticles'],passed=g['antiparticles']>0))

# Independently derive leading low-T thermal-index derivative using log-density,
# a different step and the uniform composition held fixed.
for n,Y in [(.16,.5),(.32,.1),(.8,.05)]:
    h=3e-5
    def gamma(nn):
        ss=r.cold_state(nn,Y); mass=ss.dirac_mass_mev/r.HBARC
        return sum((3*math.pi**2*y*nn)**(1/3)*math.hypot((3*math.pi**2*y*nn)**(1/3),mass) for y in [Y,1-Y])/(3*r.HBARC)
    expected=2-(math.log(gamma(n*math.exp(h)))-math.log(gamma(n*math.exp(-h))))/(2*h)
    check(f'Sommerfeld thermal index log-density derivative {n,Y}',r.sommerfeld_thermal(n,.1,Y)['thermal_index'],expected,1e-7)

spin_n,spin_T,spin_Y=.06,1.,.5
h=spin_n*2e-4
slope=(r.state(spin_n+h,spin_T,spin_Y).pressure-r.state(spin_n-h,spin_T,spin_Y).pressure)/(2*h)
checks.append(dict(name='documented spinodal independently reproduced',actual=slope,passed=slope<0))
sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(r.__file__),ROOT/'rmf_reference/validate_gm1.py']}
out=dict(passed=all(x['passed'] for x in checks),checks=checks,samples=samples,source_sha256=sources)
Path(__file__).with_name('rmf_review.json').write_text(json.dumps(out,indent=2,default=lambda x:x.item())+'\n')
print(json.dumps(dict(passed=out['passed'],checks=len(checks),failed=[x for x in checks if not x['passed']]),indent=2,default=lambda x:x.item()))
