"""Independent frozen candidate check, with original momentum/Bessel targets."""
from pathlib import Path
import sys,math,json,hashlib,importlib.util,ast,shutil,inspect
import numpy as np
import scipy.special as special
P=Path(__file__).resolve().parent;ROOT=P.parent;A=ROOT/'production_lepton_port';sys.dont_write_bytecode=True
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
expected='e0c16fc5899d5a9bfe2e8c89ba7255a60e80f0bb620c4f3103a7e4bc68487b90';manifest='26c63c1865b6dfcdc13968a78f21ab2a71ff15efd3c6fd22b19e4d52189c8e14'
assert sha(A/'staged/src/nmbackend/reference/finite_t_leptons.py')==expected
assert sha(A/'CANDIDATE_MANIFEST.json')==manifest
before={str(f.relative_to(A)):sha(f)for f in A.rglob('*')if f.is_file()}
sys.path.insert(0,str(A/'test_mirror/src'))
from nmbackend.reference import finite_t_leptons as L
assert sha(Path(L.__file__))==expected
sp=importlib.util.spec_from_file_location('nmbackend.reference.frozen_pre_bound_failure',ROOT/'review_production_lepton_compatibility/reviewed_compatibility_candidate/src/nmbackend/reference/finite_t_leptons.py');old=importlib.util.module_from_spec(sp);sys.modules[sp.name]=old;sp.loader.exec_module(old)
lock=json.loads((P/'VALIDATION_LOCK.json').read_text());checks=[];states=[]
def ck(n,e,t=3e-10):checks.append(dict(name=n,error=float(e),gate=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
def traced(state):
 integrate=L._integrate;restore=L._restore;active=[];log=[]
 def track(fun,*a,**kw):
  out=integrate(fun,*a,**kw)
  if fun.__name__=='hole_integrand':active.append(True);log.append(dict(event='hole_integrated',scaled_integral=float(out)))
  return out
 def restored(value,log_scale):
  out=restore(value,log_scale)
  if active:
   log.append(dict(event='hole_restored',value=float(out),log_scale=float(log_scale)));active.clear()
  return out
 L._integrate=track;L._restore=restored
 try:value=L.thermal_energy_density(state)
 finally:L._integrate=integrate;L._restore=restore
 return value,log
for i,r in enumerate(lock['states']):
 s=L.state_mu(r['T'],r['mu'],r['mass']);o=old.state_mu(r['T'],r['mu'],r['mass']);ck('statefields bitwise '+str(i),0 if s.asdict()==o.asdict()else 1,0)
 ck('independent FD number '+str(i),s.net_density/r['momentum_number']-1);ck('independent FD energy '+str(i),s.energy_density/r['momentum_energy']-1)
 try:v,events=traced(s);exc=None
 except ArithmeticError as e:v=None;events=[];exc=str(e)
 states.append(dict(index=i,oracle=r,value=v,events=events,exception=exc))
 ck('thermal functional completion '+str(i),0 if exc is None else 1,0)
 if v is not None:ck('independent FD thermal '+str(i),v/r['momentum_thermal']-1)
 pf=L.HBARC*(3*math.pi**2*abs(s.net_density))**(1/3)/s.temperature;xc=pf*pf/(math.hypot(pf,s.mass/s.temperature)+s.mass/s.temperature)
 B=L._hole_log2_upper_bound(s.temperature,s.mass/s.temperature,xc);states[-1]['candidate_log2_bound']=B
 # Exact represented frexp exponent bound must enclose independent tighter bound.
 ck('dyadic bound above analytic '+str(i),max(0.,r['log_upper_hole_physical']-B*math.log(2.)),2e-12)
 if B<=-1075:ck('skip only guaranteedunrepresented '+str(i),0 if r['log_upper_hole_physical']<r['log_halfminsubnormal']and not events else 1,0)
 if i<8 and r['hole_float']>0:
  holes=[x['value']for x in events if x['event']=='hole_restored'];ck('representedhole actuallyintegrated '+str(i),0 if len(holes)==1 and holes[0]>0 else 1,0)
  if holes:ck('representedhole absoluteULP '+str(i),(holes[0]-r['hole_float'])/float(np.nextafter(0.,1.)),1)
 if i>=8:
  try:old.thermal_energy_density(o);oldfail=False
  except ArithmeticError:oldfail=True
  ck('original failure reproduced keptOLDFAIL '+str(i),0 if oldfail else 1,0);states[-1]['old_thermal_failure_reproduced']=oldfail
for i,r in enumerate(lock['additional_valid_states']):
 o=old.state_mu(r['T'],r['mu'],r['mass']);s=L.state_mu(r['T'],r['mu'],r['mass']);ck('freshvalidstate unchanged '+str(i),0 if o.asdict()==s.asdict()else 1,0);ck('freshvalidthermal unchanged '+str(i),L.thermal_energy_density(s)/old.thermal_energy_density(o)-1,3e-12)
# Strong analytic invariant at independently selected exponent stress controls.
for T,v,xc in[(2.**-250,0.,2.**-10),(2.**-200,3.,2.**-40),(2.**80,1e100,1e-150),(1e-3,500.,1e-170),(.7,3.2,.013)]:
 B=L._hole_log2_upper_bound(T,v,xc);tight=4*math.log(T)+math.log(1/(math.pi**2*L.HBARC**3))+math.log(4/15)+.5*math.log(xc+2*v)+math.log(v+xc)+2.5*math.log(xc)
 ck('frexp enclosurenewcontrol '+str((T,v,xc)),max(0.,tight-B*math.log(2.)),2e-12)
# Existing minimum-API helper behavior and real package import with newer export absent.
for x in[-1e300,-803.,-29.4,-.51,-1e-20,-0.,0.,1e-20,.51,29.4,803.,1e300]:
 ref=-math.log1p(math.exp(-x))if x>=0 else x-math.log1p(math.exp(x));ck('privatehelper scalar '+str(x),(float(L._log_expit(x))-ref)/max(abs(ref),1e-300),2e-15)
saved=special.log_expit
try:
 del special.log_expit
 q=importlib.util.spec_from_file_location('nmbackend.reference.thermal_bound_missing_export',Path(L.__file__));fresh=importlib.util.module_from_spec(q);sys.modules[q.name]=fresh;q.loader.exec_module(fresh)
 ck('realrelativeimport missingnewSciPyexport',0 if fresh.thermal_energy_density(fresh.state_mu(.7,.23))>0 else 1,0)
finally:special.log_expit=saved
for name in['state_mu','state_density','thermal_energy_density']:ck('publicsignatureunchanged '+name,0 if inspect.signature(getattr(old,name))==inspect.signature(getattr(L,name))else 1,0)
ot=ast.parse(Path(old.__file__).read_text());nt=ast.parse(Path(L.__file__).read_text());of={n.name:ast.dump(n,include_attributes=False)for n in ot.body if isinstance(n,ast.FunctionDef)};nf={n.name:ast.dump(n,include_attributes=False)for n in nt.body if isinstance(n,ast.FunctionDef)}
for n in of:
 if n!='thermal_energy_density':ck('unchangedAST '+n,0 if of[n]==nf[n]else 1,0)
# Thermally changed function differs only by extra conjunction in original hole branch.
therm=next(n for n in nt.body if isinstance(n,ast.FunctionDef)and n.name=='thermal_energy_density')
for node in ast.walk(therm):
 if isinstance(node,ast.If)and isinstance(node.test,ast.BoolOp)and isinstance(node.test.op,ast.And)and any(isinstance(x,ast.Compare)and isinstance(x.left,ast.Call)and isinstance(x.left.func,ast.Name)and x.left.func.id=='_hole_log2_upper_bound'for x in node.test.values):node.test=node.test.values[0]
ck('thermalASTonlyboundconjunction',0 if ast.dump(therm,include_attributes=False)==of['thermal_energy_density']else 1,0)
for name,h in json.loads((A/'CANDIDATE_MANIFEST.json').read_text())['files'].items():ck('candidatepath '+name,0 if sha(A/'staged'/name)==h else 1,0)
after={str(f.relative_to(A)):sha(f)for f in A.rglob('*')if f.is_file()};ck('allauthorbytesunchanged',0 if before==after else 1,0);ck('initialfailedreviewunchanged',0 if sha(ROOT/'review_production_lepton_compatibility/REVIEW.md')=='a3aa12992631504f83811ec569ac9fc044b91246f2cbe5325ed6733e94430f43'else 1,0)
out=dict(checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],states=states,source_sha256=expected,manifest_sha256=manifest,authorbytes_unchanged=before==after)
(P/'results.json').write_text(json.dumps(out,indent=2)+'\n');print(out['passed'],out['failed'])
