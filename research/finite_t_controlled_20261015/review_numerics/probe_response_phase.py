"""Independent small-q transition response and RMF Hessian review."""
from pathlib import Path
import hashlib,json,math,sys
import numpy as np
from scipy.integrate import quad
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'response_reference'),str(ROOT/'rmf_phase'),str(ROOT/'rmf_reference')]
import free_density_response as d
import stability as st
import gm1_finite_t as r
checks=[];samples=[]
def check(name,actual,expected,tol=1e-5,scale=1e-25):
    error=float(np.linalg.norm(np.asarray(actual)-expected)/max(np.linalg.norm(np.asarray(expected)),scale))
    checks.append(dict(name=name,actual=np.asarray(actual).tolist(),expected=np.asarray(expected).tolist(),relative_error=error,passed=error<tol))

# Independent energy-space FD product after integrating the longitudinal delta.
# Normalize a scaled scalar integral before comparing logarithms.
for a in [-100.,-5.,0.,10.,1000.]:
    for z in [-100.,-1e-5,-1e-8,0.,1e-8,1e-5,100.]:
        from scipy.special import log_expit
        peak=max(0.,a-z/2)
        scale=float(log_expit(a-peak)+log_expit(peak-a+z))
        f=lambda x:math.exp(float(log_expit(a-x)+log_expit(x-a+z))-scale)
        upper=max(0.,a,a-z)+80
        points=sorted({x for x in [a,a-z,peak] if 0<x<upper})
        value=quad(f,0.,upper,points=points,epsabs=1e-12,epsrel=2e-11,limit=300)[0]
        check(f'log integral a{a} z{z}',d._log_integral(a,z),scale+math.log(value),1e-9,1.)

# Small-q neutral response sums: integrate omega=q*v to resolve the narrowing
# distribution and use KMS folding for the f-sum, avoiding odd cancellation.
for T,nu in [(3.,30.),(.01,50.)]:
    s=d.PairState(T,939.,0.,0.,nu,nu)
    mass=s.mass;vF=math.sqrt(2*nu/mass)
    xmax=math.sqrt(2*(nu+80*T)/mass)
    pmax=math.sqrt(2*mass*(nu+80*T));pf=math.sqrt(2*mass*nu)
    ppoints=sorted({math.sqrt(2*mass*e) for e in [nu-40*T,nu-8*T,nu,nu+8*T,nu+40*T] if 0<e<nu+80*T})
    density=quad(lambda p:p*p*expit((nu-p*p/(2*mass))/T),0,pmax,points=ppoints,epsabs=1e-12,epsrel=1e-11)[0]/(math.pi**2*d.HBARC**3)
    susceptibility=quad(lambda p:p*p*expit((nu-p*p/(2*mass))/T)*expit((p*p/(2*mass)-nu)/T)/T,0,pmax,points=ppoints,epsabs=1e-12,epsrel=1e-10)[0]/(math.pi**2*d.HBARC**3)
    for q in [1e-4,.1]:
        width=T/(mass*vF)
        vpoints=sorted({vF+sgn*q/(2*mass)+k*width for sgn in [-1,1] for k in [-40,-8,0,8,40] if 0<vF+sgn*q/(2*mass)+k*width<xmax})
        static=quad(lambda v:q*(d.response(s,q,q*v)+d.response(s,q,-q*v))/(2*math.pi),0,xmax,points=vpoints,epsabs=1e-15,epsrel=1e-9,limit=300)[0]
        fsum=quad(lambda v:q*q*v*d.response(s,q,q*v)*(-math.expm1(-q*v/T))/(2*math.pi),0,xmax,points=vpoints,epsabs=1e-25,epsrel=1e-9,limit=300)[0]
        check(f'smallq f-sum T{T} q{q}',fsum,density*q*q/(2*mass),2e-8)
        if q==1e-4:
            check(f'smallq static compressibility T{T}',static,T*susceptibility,3e-6)
        for v in [-.4,0.,.2,.5]:
            check(f'smallq KMS T{T} q{q} v{v}',d.log_response(s,q,-q*v)-d.log_response(s,q,q*v),-q*v/T,1e-8,1.)

s=d.PairState(2.,800.,700.,715.,720.,719.)
shift=d.PairState(s.T,s.mass,s.offset_a,s.offset_b+37,s.mu_a,s.mu_b+37)
check('independent final species zero-shift convention',d.log_response(s,40,3),d.log_response(shift,40,40),1e-12)

for n,T,Y in [(.08,10.,.3),(.1,12.,.2),(.4,40.,.2)]:
    s=r.state(n,T,Y);H,detail=st.implicit_hessian(s)
    result=st.classify(s,H,detail);samples.append(result)
    pair=np.array([(1-Y)*n,Y*n]); step=1.5e-3*pair
    def at(pair_):return r.state(float(sum(pair_)),T,float(pair_[1]/sum(pair_)),initial=s)
    Hmu=np.zeros((2,2));Henergy=np.zeros((2,2));dphi=np.zeros(2)
    for j in range(2):
        vec=np.zeros(2);vec[j]=step[j];a=at(pair-vec);b=at(pair+vec)
        Hmu[:,j]=(np.array([b.mu_n_mev,b.mu_p_mev])-np.array([a.mu_n_mev,a.mu_p_mev]))/(2*step[j])
        Henergy[j,j]=(b.free_energy_density-2*s.free_energy_density+a.free_energy_density)/(step[j]**2)
        dphi[j]=(b.phi_mev-a.phi_mev)/(2*step[j]*r.HBARC)
    h,k=step
    Henergy[0,1]=Henergy[1,0]=(at(pair+[h,k]).free_energy_density-at(pair+[h,-k]).free_energy_density-at(pair+[-h,k]).free_energy_density+at(pair+[-h,-k]).free_energy_density)/(4*h*k)
    check(f'implicitH vs independent mu FD {n,T,Y}',H,Hmu,1e-5)
    check(f'implicitH vs independent freeenergy FD {n,T,Y}',H,Henergy,1e-4)
    check(f'scalar field species response {n,T,Y}',detail['dphi_dn_fm2'],dphi,1e-5)
    check(f'chemical FD raw symmetry {n,T,Y}',Hmu,Hmu.T,1e-5)
    # Scalar curvature: fixed-density inversions, distinct kinetic+field energy.
    phi=s.phi_mev/r.HBARC;mass0=939./r.HBARC;hp=5e-5*mass0
    def fphi(x):
        a=r.fixed_density_gas(T,pair[0],mass0-x)
        b=r.fixed_density_gas(T,pair[1],mass0-x)
        return (a['energy']+b['energy']+r.scalar_potential(x))*r.HBARC-T*(a['entropy']+b['entropy'])
    curvature=(fphi(phi+hp)-2*fphi(phi)+fphi(phi-hp))/(hp*hp*r.HBARC)
    check(f'canonical scalar curvature units/FD {n,T,Y}',detail['canonical_scalar_curvature_fm_minus2'],curvature,2e-5)
    if n==.08:
        checks.append(dict(name='positive fixedY slope does not prove full stability',actual=result['mechanical_dP_dn_fixed_T_Yp'],minimum_eigenvalue=result['eigenvalues_mev_fm3'][0],passed=result['mechanical_dP_dn_fixed_T_Yp']>0 and result['eigenvalues_mev_fm3'][0]<0))

files=[Path(d.__file__),Path(st.__file__),Path(r.__file__)]
out=dict(passed=all(c['passed'] for c in checks),checks=checks,phase_samples=samples,sources_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
Path(__file__).with_name('response_phase_review.json').write_text(json.dumps(out,indent=2,default=lambda x:x.item())+'\n')
print(json.dumps(dict(passed=out['passed'],checks=len(checks),failed=[c for c in checks if not c['passed']]),indent=2,default=lambda x:x.item()))
