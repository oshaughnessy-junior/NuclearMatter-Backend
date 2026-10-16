"""Separate static-compatibility recheck and representable thermal-total completion gate."""
from pathlib import Path
import sys,math,json,hashlib,importlib.util,ast,copy
import numpy as np
import scipy.special as special
from scipy.special import kve
P=Path(__file__).resolve().parent;ROOT=P.parent;A=ROOT/'production_lepton_port';sys.dont_write_bytecode=True;sys.path.insert(0,str(A/'test_mirror/src'))
from nmbackend.reference import finite_t_leptons as L
from nmbackend.constants import M_E,M_MU,HBARC
checks=[];functional=[];rows=[];before={str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest()for q in A.rglob('*')if q.is_file()}
def ck(n,e,t=3e-10):checks.append(dict(name=n,error=float(e),gate=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
def load(n,p):
 sp=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(sp);sys.modules[n]=m;sp.loader.exec_module(m);return m
old=load('nmbackend.reference.preserved_scipy18_before_compat',A/'pre_scipy17_repair/staged/src/nmbackend/reference/finite_t_leptons.py')
ck('new source SHA',0 if hashlib.sha256(Path(L.__file__).read_bytes()).hexdigest()=='53518a39d9b4030745285d774e028e8814485ca25d48f8fa45587d5a14e9af9e'else 1,0)
ck('new manifest SHA',0 if before['production_lepton_port/CANDIDATE_MANIFEST.json']=='bdb62cfe3e4192d583d648315ffeaa8219eb1a2d0fea4621396b6c3a4714bb0d'else 1,0)
for x in [-1e300,-1000.,-73.1,-.37,-1e-20,-0.,0.,1e-20,.37,73.1,1000.,1e300]:
 ref=-math.log1p(math.exp(-x))if x>=0 else x-math.log1p(math.exp(x));v=float(L._log_expit(x));ck('minimum-compatible helper '+repr(x),(v-ref)/max(abs(ref),1e-300),2e-15)
# Every old function is AST-identical after the private helper rename.
class Rename(ast.NodeTransformer):
 def visit_Name(self,n):
  if n.id=='_log_expit':n.id='log_expit'
  return n
oldtree=ast.parse((A/'pre_scipy17_repair/staged/src/nmbackend/reference/finite_t_leptons.py').read_text());newtree=Rename().visit(ast.parse(Path(L.__file__).read_text()));of={n.name:ast.dump(n,include_attributes=False)for n in oldtree.body if isinstance(n,ast.FunctionDef)};nf={n.name:ast.dump(n,include_attributes=False)for n in newtree.body if isinstance(n,ast.FunctionDef)and n.name!='_log_expit'}
ck('all numerical/public-function AST unchanged',0 if of==nf else 1,0)
# The missing newer export now permits a fresh actual package-relative import.
saved=special.log_expit
try:
 del special.log_expit;fresh=load('nmbackend.reference.compat_fresh_missing_export',Path(L.__file__));s=fresh.state_mu(1.37,-.29);ck('missing SciPy1.8 export fresh import works',0 if s.energy_density>0 else 1,0)
finally:special.log_expit=saved
for T,mu,m in[(.87,-1.31,M_E),(17.8,93.7,M_MU),(.01,151.3,M_MU),(M_E/40,0.,M_E),(M_E/600,0.,M_E)]:
 a=old.state_mu(T,mu,m);b=L.state_mu(T,mu,m)
 for k,v in a.asdict().items():ck('old/new selected field '+str((T,mu,k)),float(getattr(b,k))/float(v)-1 if v else float(getattr(b,k)),3e-12 if v else 0)
 for mod,st in [(old,a),(L,b)]:
  val=mod.thermal_energy_density(st);rows.append(dict(T=T,mu=mu,m=m,module=mod.__name__,thermal=val))
 ck('old/new successful thermal '+str((T,mu)),rows[-1]['thermal']/rows[-2]['thermal']-1,3e-12)
for mass,T in[(M_E,.93),(M_MU,18.7)]:
 for n in [-6.1e-29,6.1e-29]:
  s=L.state_density(T,n,mass);ck('new signed root '+str((mass,n)),s.net_density/n-1,1e-9)
# Independent physical forward target for the known returned-state gap.
for alpha in [-.53,.53]:
 T=M_E/600;mu=alpha*T;s=L.state_mu(T,mu);z=M_E/T;base=M_E*M_E*T/(math.pi**2*HBARC**3)*kve(2,z)*math.exp(-z);n=2*base*math.sinh(alpha);mean=3*T+M_E*kve(1,z)/kve(2,z);energy=2*base*math.cosh(alpha)*mean;expected=energy-M_E*abs(n)
 ck('representable thermal independent state energy '+str(alpha),s.energy_density/energy-1,3e-10);ck('representable thermal independent net '+str(alpha),s.net_density/n-1,3e-10)
 outcomes={}
 for name,mod in [('old',old),('new',L)]:
  try:value=mod.thermal_energy_density(mod.state_mu(T,mu));outcomes[name]=dict(value=value,error=value/expected-1,exception=None)
  except ArithmeticError as exc:outcomes[name]=dict(value=None,exception=str(exc))
 pf=HBARC*(3*math.pi**2*abs(n))**(1/3)/T;v=M_E/T;xc=pf*pf/(math.hypot(pf,v)+v)
 # Rigorous upper bound using vacancy<=1 and measure<=sqrt(x)*sqrt(xc+2v)*(v+xc).
 log_dimless_hole=math.log(4/15)+.5*math.log(xc+2*v)+math.log(v+xc)+2.5*math.log(xc);log_physical_hole=log_dimless_hole+4*math.log(T)-math.log(math.pi**2*HBARC**3)
 ok=outcomes['new']['exception']is None and abs(outcomes['new']['error'])<3e-8
 functional.append(dict(name='returned-state representable thermal completion '+str(alpha),passed=bool(ok),T=T,mu=mu,mass=M_E,expected=expected,expected_is_normal=bool(expected>np.finfo(float).tiny),public_outcomes=outcomes,cold_kinetic_edge=xc,log_upper_hole_dimensionless=log_dimless_hole,log_upper_hole_physical=log_physical_hole,smallest_float_log=math.log(float(np.nextafter(0.,1.)))))
# Verify unchanged historical failures and author candidate, not self-acceptance.
for dirn,pinname in [('review_production_lepton_repair','INDEPENDENT_QUADRATURE_REVIEW_PIN.json'),('review_production_lepton_port','INITIAL_INDEPENDENT_REVIEW_PIN.json')]:
 for name,h in json.loads((A/pinname).read_text()).items():ck('prior frozen review '+dirn+'/'+name,0 if hashlib.sha256((ROOT/dirn/name).read_bytes()).hexdigest()==h else 1,0)
for name,h in json.loads((A/'CANDIDATE_MANIFEST.json').read_text())['files'].items():ck('five candidate path '+name,0 if hashlib.sha256((A/'staged'/name).read_bytes()).hexdigest()==h else 1,0)
after={str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest()for q in A.rglob('*')if q.is_file()};ck('all author bytes unchanged',int(before!=after),0)
out=dict(checks=checks,passed=sum(c['passed']for c in checks),numerical_failed=[c for c in checks if not c['passed']],functional_checks=functional,functional_failed=[c for c in functional if not c['passed']],rows=rows,before=before,after=after);(P/'results.json').write_text(json.dumps(out,indent=2)+'\n');print(out['passed'],out['numerical_failed'],'FUNCTIONAL_FAILED',len(out['functional_failed']));print(functional)
