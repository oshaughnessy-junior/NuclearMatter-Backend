"""Safe selected-function reuse: no upstream driver/import aliases or factorycalls."""
from pathlib import Path
from types import SimpleNamespace,ModuleType
import ast,copy,math,json,sys,time,hashlib
from dataclasses import dataclass
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import expit,log_expit
from scipy.optimize import brentq
P=Path(__file__).resolve().parent;R=P.parent;LOCK=json.loads((P/'PREPARATION_LOCK.json').read_text());MN=939.56542052;MP=938.27208816;ME=.51099895069;HC=197.3269804;HBAR=6.582119569e-22;MW=(MN+MP)/2
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for rel,h in LOCK['sourcepins'].items():
 if sha(R/rel)!=h:raise RuntimeError('Frozenoracledependencychanged '+rel)
def nodes(rel,names):return [copy.deepcopy(n)for n in ast.parse((R/rel).read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in names]
def module(name,body,env):
 m=ModuleType(name);m.__file__=str(P/(name+'.py'));sys.modules[name]=m;m.__dict__.update(dict(np=np,math=math,json=json,time=time,leggauss=leggauss,expit=expit,log_expit=log_expit,brentq=brentq,dataclass=dataclass));m.__dict__.update(env)
 tree=ast.fix_missing_locations(ast.Module(body=body,type_ignores=[]));(P/(name+'.py')).write_text('# Generatedselecteddefinitions; environmentexplicitinoracles.py; notupstreamdriver.\n'+ast.unparse(tree)+'\n');exec(compile(tree,m.__file__,'exec'),m.__dict__);return m
# PureDiracprimitives: const/gamma setup plus functiondefinitions, no path/IO.
epath=R/'relativistic_capture_oracle/events.py';etree=ast.parse(epath.read_text());ebody=[]
for n in etree.body:
 if (isinstance(n,ast.Assign)and all(isinstance(t,ast.Name)and t.id in {'GF','VUD','HBAR','HC','S','Z','G','G5','I','MET'}for t in n.targets)) or isinstance(n,ast.FunctionDef)and n.name in ['spinors','adj','vertices','boost']:
  ebody.append(copy.deepcopy(n))
E=module('weak_review_dirac_primitives',ebody,{})
# Source scalarinvariants provider: no originalgenericcurrent/vacuum imports.
V=module('weak_review_vacuum_scalar_primitives',nodes('muon_vacuum/vacuum_cc.py',['Channel','dot']),dict(MN=MN,MP=MP,ME=ME,HBARC=HC,GF=1.1663785e-11,VUD=.974,METRIC=np.array([1.,-1.,-1.,-1.])))
C=module('weak_review_source_current',nodes('muon_full_current/current.py',['FormFactors','guo_invariants']),dict(V=V))
# EC different invariant-s / original-parentmomentum / boost-axis / directtrace.
ecbody=nodes('review_hot_electron_capture/probe.py',['dot','slash','trace_amp','integrate'])
ecfn=next(n for n in ecbody if n.name=='integrate');ecfn.args.args.append(ast.arg(arg='radiation_mus'));ecfn.args.defaults.append(ast.Constant(None))
for n in ast.walk(ecfn):
 if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='scenarios'for t in n.targets):n.value=ast.IfExp(test=ast.Compare(left=ast.Name('radiation_mus',ast.Load()),ops=[ast.IsNot()],comparators=[ast.Constant(None)]),body=ast.Name('radiation_mus',ast.Load()),orelse=n.value)
EC=module('weak_review_ec_phase_trace',ecbody,dict(E=E,C=SimpleNamespace(MN=MN,MP=MP,ME=ME,MW=MW,HC=HC,HBAR=HBAR)))
# PC independentreviewerpolynomial staysmathematicallyunchanged; radiations/entropyonly.
pcbody=nodes('review_pc_moments/reference.py',['dot','boost','amplitude','integral']);pcfn=next(n for n in pcbody if n.name=='integral');pcfn.args.args.append(ast.arg(arg='radiation_mus'));pcfn.args.defaults.append(ast.Constant(None))
for n in ast.walk(pcfn):
 if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='scenarios'for t in n.targets):n.value=ast.IfExp(test=ast.Compare(left=ast.Name('radiation_mus',ast.Load()),ops=[ast.IsNot()],comparators=[ast.Constant(None)]),body=ast.Name('radiation_mus',ast.Load()),orelse=n.value)
 if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='zeros'and n.args:
  dim=n.args[0]
  if isinstance(dim,ast.Tuple)and isinstance(dim.elts[0],ast.Constant)and dim.elts[0].value==3:dim.elts[0]=ast.Call(func=ast.Name('len',ast.Load()),args=[ast.Name('scenarios',ast.Load())],keywords=[])
  elif isinstance(dim,ast.Constant)and dim.value==3:n.args[0]=ast.Call(func=ast.Name('len',ast.Load()),args=[ast.Name('scenarios',ast.Load())],keywords=[])
# Replaceonlytheentropy-affinityfactorwithdirectactualF/B logratio.
for n in ast.walk(pcfn):
 if isinstance(n,ast.AugAssign)and isinstance(n.target,ast.Subscript)and isinstance(n.target.value,ast.Name)and n.target.value.id=='entropy':n.value=ast.parse('float(np.sum((WF-WB)*(np.log(F)-np.log(B))))',mode='eval').body
PC=module('weak_review_pc_phase_polynomial',pcbody,dict(MN=MN,MP=MP,ME=ME,MW=MW,GF=1.1663785e-11,VUD=.974,HC=HC,HBAR=HBAR,NAMES=['number','neutrino_energy','positron_energy','proton_energy','neutron_energy','material_energy_loss']))
# Beta flatDalitz andelectron-axisHaar, primaryscalar current, directparentp.
dbody=nodes('review_crossed_moments/reference.py',['events','integrate']);dfn=next(n for n in dbody if n.name=='integrate');dfn.args.args.extend([ast.arg(arg='radiation_mus'),ast.arg(arg='parent_tail')]);dfn.args.defaults.extend([ast.Constant(None),ast.Constant(80.)])
for n in ast.walk(dfn):
 if isinstance(n,ast.comprehension)and isinstance(n.iter,ast.List)and [x.value for x in n.iter.elts if isinstance(x,ast.Constant)]==['unblocked','empty','FD','LTE']:
  n.iter=ast.parse("['unblocked']+list(radiation_mus)",mode='eval').body
 if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='chemical'for t in n.targets):n.value=ast.Subscript(value=ast.Name('radiation_mus',ast.Load()),slice=ast.Name('name',ast.Load()),ctx=ast.Load())
 if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='sqrt'and len(n.args)==1 and isinstance(n.args[0],ast.Constant)and n.args[0].value==80.:n.args[0]=ast.Name('parent_tail',ast.Load())
BETA=module('weak_review_beta_flatdalitz',dbody,dict(C=C,MN=MN,MP=MP,ME=ME,HC=HC,HBAR=HBAR,NAMES=['number','neutrino_energy','electron_energy','proton_energy','neutron_energy','material_energy_loss']))
# ASTproof: allcurrent/event/traceamplitudes areliteralunchangedselectedfunctions.
proof={}
for name,body,rel,funcs in [('EC',ecbody,'review_hot_electron_capture/probe.py',['dot','slash','trace_amp']),('PC',pcbody,'review_pc_moments/reference.py',['dot','boost','amplitude']),('BETA',dbody,'review_crossed_moments/reference.py',['events'])]:
 original={n.name:ast.dump(n,include_attributes=False)for n in nodes(rel,funcs)}
 proof[name]={fn:ast.dump(next(n for n in body if n.name==fn),include_attributes=False)==original[fn]for fn in funcs}
assert all(all(x.values())for x in proof.values());(P/'EXTRACTION_PROVENANCE.json').write_text(json.dumps(dict(sourcepins=LOCK['sourcepins'],unchanged_AST=proof,import_policy='Onlyselecteddefinitions, scopedglobals explicitlyinjected. NOupstreamtop-leveldriver/sys.path/globalalias/filewrites executed.',intentional_integrator_changes=LOCK['changes_to_reused_purefunctions']),indent=2)+'\n')
