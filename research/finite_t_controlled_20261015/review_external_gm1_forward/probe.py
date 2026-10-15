"""Independent rapidity-coordinate warm GM1 roots; NOT native O2scl oracle."""
from pathlib import Path
import sys,math,json,hashlib,importlib.util,time
import numpy as np
from scipy.integrate import quad_vec
from scipy.optimize import root
from scipy.special import expit
P=Path(__file__).parent;ROOT=P.parent;LEAF=ROOT/'external_gm1_forward';sys.dont_write_bytecode=True;checks=[];start=time.monotonic()
def ck(name,e,tol):checks.append(dict(name=name,error=float(e),tolerance=tol,passed=bool(math.isfinite(e)and abs(e)<=tol)))
paths=[x for d in ['external_gm1_forward','rmf_reference']for x in(ROOT/d).rglob('*')if x.is_file()and'__pycache__'not in str(x)]
def snap():return{str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest()for x in paths}
before=snap();ck('commissioned manifest exact',float(hashlib.sha256((LEAF/'MANIFEST.json').read_bytes()).hexdigest()!='e07eb96cdd60b4eda2fdfdb674685cca535a840632ba22a23704921bb20811b9'),0)
for name,sha in json.loads((LEAF/'MANIFEST.json').read_text()).items():ck('author frozen manifest '+name,float(hashlib.sha256((LEAF/name).read_bytes()).hexdigest()!=sha),0)
HC=197.3269804;M=939.;Cs,Cw,Cr=11.785,7.148,4.410;b,c=.002947,-.001070
# Own MeV-unit integrals in y=asinh(p/m), independent of both authored reductions.
def gas(T,m,nu,tail=60.,rtol=2e-11):
 end=max(m,abs(nu))+tail*T;ym=math.acosh(end/m);yf=math.acosh(nu/m)if nu>m else 0.;pieces=sorted(set([0.,yf,ym]));out=np.zeros(6)
 def f(y):
  p=m*math.sinh(y);e=m*math.cosh(y);measure=p*p*e/(math.pi**2*HC**3);x=(e-nu)/T;xb=(e+nu)/T;fp=expit(-x);fb=expit(-xb)
  def ent(z):return np.logaddexp(0,-abs(z))+abs(z)*expit(-abs(z))
  return measure*np.array([fp-fb,m/e*(fp+fb),e*(fp+fb),p*p/(3*e)*(fp+fb),ent(x)+ent(xb),fb])
 for lo,hi in zip(pieces[:-1],pieces[1:]):
  value,err,info=quad_vec(f,lo,hi,epsabs=1e-13,epsrel=rtol,full_output=True)
  if not info.success:raise ArithmeticError('Rapidity quadrature failed')
  out+=value
 return out

def solve(nB,T,Y,tail=60.,rtol=2e-11):
 targets=np.array([nB*(1-Y),nB*Y]);phi0=.35*M;m0=M-phi0;pf=(3*math.pi**2*targets)**(1/3)*HC;guess=np.r_[phi0,np.sqrt(m0*m0+pf*pf)]
 def eq(x):
  phi,nn,pp=x;m=M-phi
  if not 0<m<M:raise ValueError('Outside local positive-mass reference branch')
  gn=gas(T,m,nn,tail,rtol);gp=gas(T,m,pp,tail,rtol);field=phi/(Cs*HC)+b*M*phi*phi/HC**3+c*phi**3/HC**3
  return np.r_[(field-gn[1]-gp[1])/nB,(gn[0]-targets[0])/targets[0],(gp[0]-targets[1])/targets[1]]
 ans=root(eq,guess,method='hybr',options=dict(xtol=2e-10));res=eq(ans.x)
 if not ans.success or max(abs(res))>2e-9:raise ArithmeticError('Rapidity independent root failed')
 phi,nun,nup=ans.x;m=M-phi;gn=gas(T,m,nun,tail,rtol);gp=gas(T,m,nup,tail,rtol);W=HC*Cw*nB;R=HC*Cr*nB*(2*Y-1)/2;U=phi*phi/(2*Cs*HC)+b*M*phi**3/(3*HC**3)+c*phi**4/(4*HC**3);V=HC*(Cw*nB*nB/2+Cr*(targets[1]-targets[0])**2/8)
 return dict(nB=nB,T=T,Yp=Y,phi_mev=phi,dirac_mass_mev=m,mu_n_mev=nun+W-R/2,mu_p_mev=nup+W+R/2,omega_mev=W,rho_mev=R,energy_density=gn[2]+gp[2]+U+V,pressure=gn[3]+gp[3]-U+V,entropy_density=gn[4]+gp[4],free_energy_density=gn[2]+gp[2]+U+V-T*(gn[4]+gp[4]),scalar_density=gn[1]+gp[1],antiparticle_density=gn[5]+gp[5],nn=gn[0],np=gp[0],gas_n=gn.tolist(),gas_p=gp.tolist(),U=U,V=V,residual=res.tolist(),nfev=ans.nfev)
# Fresh local branch, different from both published authored states.
state=(.19,14.,.22);own=solve(*state,tail=55.);fine=solve(*state,tail=85.,rtol=2e-13)
def load(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
A=load('review_momentum_gm1_author',LEAF/'momentum_reduction.py');I=load('review_internal_gm1_free',ROOT/'rmf_reference/gm1_finite_t.py');a=A.solve(*state,order=224,tail=70.);internal=I.state(*state).asdict()
keys=['phi_mev','dirac_mass_mev','mu_n_mev','mu_p_mev','omega_mev','rho_mev','energy_density','pressure','entropy_density','free_energy_density','scalar_density','antiparticle_density']
for k in keys:
 ck('own rapidity convergence '+k,own[k]/fine[k]-1,2e-9);ck('heldout authored momentum forward '+k,a[k]/fine[k]-1,2e-9);ck('heldout internal forward '+k,internal[k]/fine[k]-1,2e-9)
for tag,g in [('neutron',fine['gas_n']),('proton',fine['gas_p'])]:ck('scalar trace E−3P='+tag,(g[2]-3*g[3])/(fine['dirac_mass_mev']*g[1])-1,3e-12)
ck('rho half-coupling energy',fine['rho_mev']*(fine['np']-fine['nn'])/4/(HC*Cr*(fine['np']-fine['nn'])**2/8)-1,3e-12)
ck('vector chemical work twice field energy',(fine['nn']*(fine['omega_mev']-fine['rho_mev']/2)+fine['np']*(fine['omega_mev']+fine['rho_mev']/2))/(2*fine['V'])-1,3e-12)
identity=fine['energy_density']+fine['pressure']-state[1]*fine['entropy_density']-fine['nn']*fine['mu_n_mev']-fine['np']*fine['mu_p_mev'];ck('heldout Euler identity',identity/fine['energy_density'],2e-10)
# Literal source audit and unexecuted fixture disclosure, not a fabricated native run.
fixture=json.loads((LEAF/'fixture.json').read_text());raw=json.loads((LEAF/'results.json').read_text());native=(LEAF/'o2scl_fixture.cpp').read_text();src=(LEAF/'sources/eos_had_rmf.cpp').read_text();pair=(LEAF/'sources/fermion_rel.h').read_text()
for tag,ok in [('native status null',fixture['native_driver_execution']is None),('external code not executed',raw['availability']['external_execution']is False),('missing header compile retained',raw['availability']['compile_status']!=0 and'o2scl/eos_had_rmf.h'in raw['availability']['compile_stderr']),('explicitcommonM',native.count('939./hc')>=3),('zero extra couplings',all(('eos.'+x)in native for x in fixture['candidate_config']['zero_fields'])),('correctpair source calls','fet->pair_mu(ne,temper)'in src and'fet->pair_mu(pr,temper)'in src),('correctscalar b mass','duds=b*ne.m' in src),('netpairvector density','f.n-=antip.n' in pair),('pairenergy additive','f.ed+=antip.ed' in pair),('no driver internal converged field','set_fields('not in native)]:ck(tag,float(not ok),0)
ck('all frozen source/raw unchanged',float(before!=snap()),0)
r=dict(checks=checks,passed=sum(x['passed']for x in checks),failed=[x for x in checks if not x['passed']],heldout=state,rapidity_coarse=own,rapidity_fine=fine,authored_momentum=a,internal=internal,frozen_before=before,frozen_after=snap(),seconds=time.monotonic()-start,scope='Same-equation independent numerical/convention audit only. Native externalO2scl gate remainsOPEN. No derivatives/globalphase/weakrates.')
(P/'results.json').write_text(json.dumps(r,indent=2));print(json.dumps(dict(passed=r['passed'],failed=r['failed'],seconds=r['seconds'],heldout=fine),indent=2))
