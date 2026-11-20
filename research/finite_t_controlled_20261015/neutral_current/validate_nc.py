"""Primary-source elastic normalization, KMS and closed-grid structure gates."""
from pathlib import Path
import hashlib,json,math,platform,sys,time
import numpy as np
import scipy
from scipy.special import expit
from scipy.integrate import quad_vec
from free_nc import *

ROOT=Path(__file__).resolve().parent
start=time.monotonic();checks=[];samples=[];grid_tests=[]
def check(name,a,b,tol,scale=1e-300):
    err=abs(a-b)/max(abs(b),scale)
    checks.append(dict(name=name,actual=float(a),expected=float(b),scaled_error=float(err),tolerance=tol,passed=bool(err<tol)))

def literature_static(E,kind):
    # Burrows/Reddy/Thompson Eqs9,14,18,20,23: separately normalized sigma0.
    sigma0=4*GF*GF*ME*ME*HBARC**2*1e-26/math.pi
    if kind=='neutron':
        return sigma0/4*(E/ME)**2*(1+3*GA*GA)/4,sigma0/4*(E/ME)**2*(1+5*GA*GA)/6
    return sigma0/4*(E/ME)**2*(4*SIN2_THETA_W**2-2*SIN2_THETA_W+(1+3*GA*GA)/4),sigma0/6*(E/ME)**2*((.5-2*SIN2_THETA_W)**2+5*GA*GA/4)

heavy=state_density(2.,7e-5,mass=1e7)
for E in (5.,10.,20.):
    for kind,couplings in [('neutron',NEUTRON),('proton',PROTON)]:
        num=moments(heavy,E,couplings,angle_order=40,energy_rtol=2e-10)
        expected,transport=literature_static(E,kind)
        conversion=heavy.density*1e39*LIGHT
        check(f'heavy static {kind} E{E} total',num['rate_s']/conversion,expected,2e-5)
        check(f'heavy static {kind} E{E} transport',num['angular_transport_rate_s']/conversion,transport,2e-5)
        analytic=bare_static_cross_sections(E,couplings)
        check(f'literature static exact {kind} E{E} total',analytic['total_cm2'],expected,2e-14)
        check(f'literature static exact {kind} E{E} transport',analytic['transport_cm2'],transport,2e-14)
for tag,couplings,ratio in [('vector_only',Couplings(1.,0.),2/3),('axial_only',Couplings(0.,1.),10/9)]:
    num=moments(heavy,10.,couplings,angle_order=40)
    check(tag+' independent transport ratio',num['angular_transport_rate_s']/num['rate_s'],ratio,2e-5)

for T,n in [(2.,7e-5),(2.,.002),(.5,.005)]:
    s=state_density(T,n);pair=pair_from_species(s);label=f'T={T},n={n}'
    for Ein,Eout in [(5.,4.),(10.,11.),(20.,16.),(10.,10.0000002)]:
        for c in (-.8,0.,.6):
            lf=log_kernel(pair,Ein,Eout,c);lr=log_kernel(pair,Eout,Ein,c)
            expected=2*math.log(Ein/Eout)-(Ein-Eout)/T
            check(label+f' KMS E{Ein}->{Eout} c{c}',lr-lf,expected,4e-11,1.)
            raw=kernel(pair,Ein,Eout,c)
            check(label+f' final Pauli E{Ein}->{Eout} c{c}',kernel(pair,Ein,Eout,c,final_occupation=.3),.7*raw,2e-14)
    shifted=PairState(T,s.mass,100.,100.,100.+s.kinetic_mu,100.+s.kinetic_mu)
    check(label+' constant offset cancels',log_kernel(shifted,10.,11.,.2),log_kernel(pair,10.,11.,.2),2e-13,1.)
    for E in (5.,10.,20.):
        coarse=moments(s,E,angle_order=24)
        fine=moments(s,E,angle_order=48,energy_rtol=2e-11,tail=50.)
        for key in ('rate_s','angular_transport_rate_s','energy_to_bath_MeV_s'):
            check(label+f' moment refinement E{E} '+key,coarse[key],fine[key],2e-5,1e-6*fine['rate_s'])
        blocked=moments(s,E,angle_order=40,radiation=(3.,0.))
        check(label+f' final-blocking bounds E{E}',float(0<blocked['rate_s']<=fine['rate_s']*(1+1e-10)),1.,1e-12)
        samples.append(dict(species=s.asdict(),Ein=E,unblocked=fine,blocked=blocked))
    # Explicitly closed finite grid; no transitions to unrepresented energies.
    E=np.array([2.,4.,6.,8.,10.5,14.,18.,24.]);w=np.array([2.,2.,2.,2.,3.,4.,4.,8.])
    for mu in (-5.,0.,8.):
        f=expit((mu-E)/T)
        r=discrete_collision(s,E,w,f)
        A=r['transition_rates'];g=r['density_weights']
        gross=np.sum(g[:,None]*A*f[:,None]*(1-f[None,:]))
        check(label+f' closed-grid FD stationary mu{mu}',np.max(np.abs(r['number_source'])),0.,2e-12,max(gross,1e-300))
        check(label+f' closed-grid FD number mu{mu}',r['net_number_source'],0.,2e-13,max(gross,1e-300))
        grid_tests.append(dict(T=T,n=n,mu_nu=mu,max_scaled_bin_source=float(np.max(np.abs(r['number_source']))/gross),number_source=r['net_number_source']))
    f=expit((-2.-E)/(2*T));r=discrete_collision(s,E,w,f)
    A=r['transition_rates'];g=r['density_weights'];gross=np.sum(g[:,None]*A*f[:,None]*(1-f[None,:]))
    entropy_production=-sum(r['number_source']*(np.log(f)-np.log1p(-f)+E/T))
    check(label+' nonequilibrium number conservation',r['net_number_source'],0.,2e-13,gross)
    pairflux=g[:,None]*A*f[:,None]*(1-f[None,:])
    bath_heat=float(np.sum((E[:,None]-E[None,:])*pairflux))
    check(label+' radiation vs pairwise bath energy',r['net_neutrino_energy_source']+bath_heat,0.,2e-13,gross*max(E))
    check(label+' bath+radiation entropy production positive',float(entropy_production>0),1.,1e-12)
    grid_tests.append(dict(T=T,n=n,non_equilibrium=True,number_source=r['net_number_source'],neutrino_energy_source=r['net_neutrino_energy_source'],bath_energy_source=bath_heat,total_entropy_production_fm3_s=entropy_production))

# Independent angular variable q rather than c at fixed energies.
# The transformation dc=q/(Ein*Eout)dq tests the final-state measure explicitly.
s=state_density(2.,.002);pair=pair_from_species(s)
from scipy.integrate import quad
for Ein,Eout in [(10.,9.),(10.,11.),(20.,16.)]:
    c_integral=quad(lambda c:kernel(pair,Ein,Eout,c),-1.,1.,epsabs=1e-14,epsrel=1e-10)[0]
    q_integral=quad(lambda q:kernel(pair,Ein,Eout,(Ein*Ein+Eout*Eout-q*q)/(2*Ein*Eout))*q/(Ein*Eout),abs(Ein-Eout),Ein+Eout,epsabs=1e-14,epsrel=1e-10)[0]
    check(f'independent q angle integral {Ein}->{Eout}',c_integral,q_integral,2e-9)

# Regressions for independent reviewers' input and cancellation counterexamples.
def rejects(name,call):
    try:
        call()
    except ValueError:
        check(name,1.,1.,1e-15)
    else:
        check(name,0.,1.,1e-15)

s=state_density(1.3,.0013,offset=17.)
E=np.array([3.5,5.,7.5,11.,18.]);w=np.array([.3,1.3,.8,2.1,4.])
f=np.array([.91,.07,.63,.21,.005])
for name,kwargs in [('negative tail',dict(tail=-1.)),('NaN tail',dict(tail=np.nan)),('negative radiation T',dict(radiation=(-1.,0.))),('zero radiation T',dict(radiation=(0.,0.))),('NaN radiation mu',dict(radiation=(2.,np.nan))),('infinite tolerance',dict(energy_rtol=np.inf)),('negative tolerance',dict(energy_rtol=-1.)),('noninteger angle order',dict(angle_order=2.5)),('boolean angle order',dict(angle_order=True)),('NaN incident energy',dict(Ein=np.nan))]:
    args=dict(Ein=7.,angle_order=8);args.update(kwargs)
    rejects('repair rejects '+name,lambda args=args:moments(s,**args))
for which in ('energy','weight','occupation'):
    for bad in (np.nan,np.inf):
        args=[E.copy(),w.copy(),f.copy()];args[('energy','weight','occupation').index(which)][2]=bad
        rejects(f'repair rejects {which} {bad}',lambda args=args:discrete_collision(s,*args))
rejects('repair rejects empty grid',lambda:discrete_collision(s,[],[],[]))
rejects('repair rejects invalid grid angle',lambda:discrete_collision(s,E,w,f,angle_order=0))
r=discrete_collision(s,E,w,f,Couplings(.2,.7),angle_order=40)
g,A=r['density_weights'],r['transition_rates']
flux=g[:,None]*A*f[:,None]*(1-f[None,:])
pair_entropy=0.;bath_heat=0.
for i in range(len(E)):
    for j in range(i):
        forward,backward=flux[i,j],flux[j,i]
        pair_entropy+=(forward-backward)*math.log(forward/backward)
        bath_heat+=(E[i]-E[j])*(forward-backward)
macro=-r['number_source']@(np.log(f)-np.log1p(-f)+E/s.T)
check('repair heldout irregular entropy identity',macro,pair_entropy,2e-12)
check('repair heldout paired energy identity',r['net_neutrino_energy_source'],-bath_heat,2e-12)
check('repair heldout number conservation',r['net_number_source'],0.,2e-13,float(sum(abs(r['number_source']))))
check('repair heldout positive entropy',float(macro>0),1.,1e-15)
# Replay the original column-minus-row cancellation without changing its gate.
original_dN=flux.sum(axis=0)-flux.sum(axis=1)
original_macro=-original_dN@(np.log(f)-np.log1p(-f)+E/s.T)
repair_regression=dict(original_entropy_relative_error=float(abs(original_macro/pair_entropy-1)),repaired_entropy_relative_error=float(abs(macro/pair_entropy-1)),gross_transition_flux=float(flux.sum()),original_number_source=original_dN.tolist(),repaired_number_source=r['number_source'].tolist(),pair_entropy=float(pair_entropy),macro_entropy=float(macro))

paths=['response_reference/free_density_response.py','nr_baryon_eos/free_nucleons.py','weak_rates/static_capture.py']
report=dict(passed=all(c['passed'] for c in checks),checks=checks,selected_moments=samples,closed_grid_tests=grid_tests,repair_regression=repair_regression,runtime_seconds=time.monotonic()-start,
            couplings=dict(neutron=dict(vector=NEUTRON.vector,axial=NEUTRON.axial),proton=dict(vector=PROTON.vector,axial=PROTON.axial),sin2thetaW=SIN2_THETA_W,GF_MeVminus2=GF,gA=GA),
            sources=[dict(url='https://arxiv.org/html/1801.07077',verified='2026-10-15',equations='1,2; dynamic kernel and density/spin Fourier convention'),dict(url='https://arxiv.org/html/astro-ph/0404432',verified='2026-10-15',equations='9,14,18,20,23,33; static absolute/transport and collision measure')],
            source_sha256={p:hashlib.sha256((ROOT.parent/p).read_bytes()).hexdigest() for p in paths},environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()),
            scope='Leading free unpolarized NR nuclear NC; exact quadratic phase space and Pauli blocking; selected moments and closed discrete-grid identities only; independent review required')
(ROOT/'validation_results.json').write_text(json.dumps(report,indent=2)+'\n')
print(f"{sum(c['passed'] for c in checks)}/{len(checks)} checks pass in {report['runtime_seconds']:.2f}s")
for c in checks:
    if not c['passed']:print('FAIL',c)
raise SystemExit(0 if report['passed'] else 1)
