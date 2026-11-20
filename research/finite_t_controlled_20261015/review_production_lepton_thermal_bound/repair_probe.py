"""Different independent recheck; initial104/6 artifacts are never overwritten."""
from pathlib import Path
import sys,math,json,hashlib,importlib.util,ast
import numpy as np
from scipy.integrate import quad
from scipy.special import expit
P=Path(__file__).resolve().parent;R=P.parent;A=R/'production_lepton_port';sys.dont_write_bytecode=True
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
SOURCE='baeef78b174ecc0bcc1ab6016c839a8e8dc6a2ba915541d7fd994d994f01e719';MANIFEST='5e9a2793bc0de9c2e2a01c78061bd7a76f093b260cb8a7564971aae18a1bf282'
assert sha(A/'staged/src/nmbackend/reference/finite_t_leptons.py')==SOURCE;assert sha(A/'CANDIDATE_MANIFEST.json')==MANIFEST
before={str(f.relative_to(A)):sha(f)for f in A.rglob('*')if f.is_file()};initial={str(f.relative_to(P)):sha(f)for f in P.rglob('*')if f.is_file()and not f.name.startswith('repair_')}
sys.path.insert(0,str(A/'test_mirror/src'))
from nmbackend.reference import finite_t_leptons as L
assert sha(Path(L.__file__))==SOURCE
s=importlib.util.spec_from_file_location('nmbackend.reference.frozen_initial_bound_only',P/'reviewed_candidate/staged/src/nmbackend/reference/finite_t_leptons.py');O=importlib.util.module_from_spec(s);sys.modules[s.name]=O;s.loader.exec_module(O)
lock=json.loads((P/'VALIDATION_LOCK.json').read_text());newlock=json.loads((P/'REPAIR_VALIDATION_LOCK.json').read_text());checks=[];rows=[]
def ck(n,e,t=3e-10):checks.append(dict(name=n,error=float(e),gate=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
def thermal(module,state):
 integrate=module._integrate;restore=module._restore;active=[];events=[]
 def track(fun,*args,**kw):
  try:r=integrate(fun,*args,**kw)
  except ArithmeticError as e:events.append(dict(component=fun.__name__,exception=str(e)));raise
  if fun.__name__=='hole_integrand':active.append(True);events.append(dict(event='hole_integrated',normalized=float(r)))
  return r
 def restored(value,scale):
  r=restore(value,scale)
  if active:events.append(dict(event='hole_restored',value=float(r)));active.clear()
  return r
 module._integrate=track;module._restore=restored
 try:return dict(value=module.thermal_energy_density(state),exception=None,events=events)
 except ArithmeticError as e:return dict(value=None,exception=str(e),events=events)
 finally:module._integrate=integrate;module._restore=restore
for i,r in enumerate(lock['states']):
 old=O.state_mu(r['T'],r['mu'],r['mass']);new=L.state_mu(r['T'],r['mu'],r['mass']);a=thermal(O,old);b=thermal(L,new);rows.append(dict(index=i,old=a,new=b))
 ck('stateallfieldsbitwise '+str(i),0 if old.asdict()==new.asdict()else 1,0)
 ck('independentmomentumFDn '+str(i),new.net_density/r['momentum_number']-1);ck('independentmomentumFDE '+str(i),new.energy_density/r['momentum_energy']-1)
 ck('allnewthermalcomplete '+str(i),0 if b['exception']is None else 1,0)
 if b['value']is not None:ck('newthermal independentFD '+str(i),b['value']/r['momentum_thermal']-1)
 if 4<=i<=7:ck('exactinitialmuonfailurestillOLDFAIL '+str(i),0 if a['exception']=='Unresolved finite-temperature lepton quadrature'else 1,0)
 if i<8:
  holes=[e['value']for e in b['events']if e.get('event')=='hole_restored'];ck('nearboundhole evaluated '+str(i),0 if len(holes)==1 else 1,0)
  if holes:
   ck('independenthole absoluteULP '+str(i),(holes[0]-r['hole_float'])/float(np.nextafter(0.,1.)),1)
   if r['hole_float']>0:ck('representedpositivehole notzero '+str(i),0 if holes[0]>0 else 1,0)
 if i>=8:ck('originalelectron positiveTOTAL '+str(i),0 if b['value']and b['value']>np.finfo(float).tiny else 1,0)
for i,r in enumerate(lock['additional_valid_states']+[newlock['new_regular_case']]):
 old=O.state_mu(r['T'],r['mu'],r['mass']);new=L.state_mu(r['T'],r['mu'],r['mass']);ck('freshregularstatebitwise '+str(i),0 if old.asdict()==new.asdict()else 1,0)
 a=O.thermal_energy_density(old);b=L.thermal_energy_density(new);ck('freshregulartotalpreserved '+str(i),b/a-1,3e-12)
for i,r in enumerate(newlock['new_massless_cases']):
 st=L.state_mu(r['T'],r['mu'],0.);v=thermal(L,st);T=r['T'];eta=r['mu']/T;xc=(3*math.pi**2*abs(st.net_density)*L.HBARC**3)**(1/3)/T
 holes=[e['value']for e in v['events']if e.get('event')=='hole_restored'];ck('masslesswholecompletion '+str(i),0 if v['exception']is None else 1,0)
 # Original energy-coordinate quadrature, distinct from unit-interval evaluation.
 direct=T**4/(math.pi**2*L.HBARC**3)*quad(lambda x:x*x*(xc-x)*expit(x-eta),0.,xc,epsabs=1e-100,epsrel=2e-12)[0]
 ck('masslessholecaptured '+str(i),0 if len(holes)==1 and holes[0]>0 else 1,0)
 if holes:ck('masslesshole independentoriginalenergy '+str(i),holes[0]/direct-1,3e-10)
 if i==0:
  analytic=T**4/(math.pi**2*L.HBARC**3)*xc**4/24;ck('massless exactpolynomial halfvacancy limit',direct/analytic-1,3e-10)
# Verify only the exact hole branch changes; all other syntax and dyadic gate stay.
ot=ast.parse(Path(O.__file__).read_text());nt=ast.parse(Path(L.__file__).read_text());of={x.name:x for x in ot.body if isinstance(x,ast.FunctionDef)};nf={x.name:x for x in nt.body if isinstance(x,ast.FunctionDef)}
for name,node in of.items():
 if name!='thermal_energy_density':ck('unchangedfunctionAST '+name,0 if ast.dump(node,include_attributes=False)==ast.dump(nf[name],include_attributes=False)else 1,0)
def hole_if(tree):return next(n for n in ast.walk(tree)if isinstance(n,ast.If)and isinstance(n.test,ast.BoolOp)and any(isinstance(x,ast.Compare)and isinstance(x.left,ast.Call)and isinstance(x.left.func,ast.Name)and x.left.func.id=='_hole_log2_upper_bound'for x in n.test.values))
a=hole_if(of['thermal_energy_density']);b=hole_if(nf['thermal_energy_density']);ck('dyadicconditionunchanged',0 if ast.dump(a.test,include_attributes=False)==ast.dump(b.test,include_attributes=False)else 1,0);b.body=a.body
ck('onlyholebodychanged',0 if ast.dump(of['thermal_energy_density'],include_attributes=False)==ast.dump(nf['thermal_energy_density'],include_attributes=False)else 1,0)
for name,h in json.loads((A/'CANDIDATE_MANIFEST.json').read_text())['files'].items():ck('candidatefivepath '+name,0 if sha(A/'staged'/name)==h else 1,0)
for name,h in initial.items():ck('initialreviewbytesunchanged '+name,0 if sha(P/name)==h else 1,0)
after={str(f.relative_to(A)):sha(f)for f in A.rglob('*')if f.is_file()};ck('wholeauthorbytesunchanged',0 if before==after else 1,0)
out=dict(status='ACCEPTED_SELECTED_REPAIR'if all(c['passed']for c in checks)else'BLOCKED',checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],rows=rows,source_sha256=SOURCE,manifest_sha256=MANIFEST,original_review_sha256=sha(P/'REVIEW.md'),initial_104pass6fail_preserved=sha(P/'results.json')==json.loads((P/'INITIAL_FAILURE_PIN.json').read_text())['results.json'])
(P/'repair_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out['status'],out['passed'],out['failed'])
