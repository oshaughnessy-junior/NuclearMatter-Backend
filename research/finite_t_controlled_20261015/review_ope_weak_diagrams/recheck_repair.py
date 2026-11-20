"""Independent repair recheck; never imports/replays the original review driver."""
from __future__ import annotations
from pathlib import Path
import importlib.util,importlib.machinery,sys,json,hashlib,ast,math,time
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;AUTHOR=ROOT/'ope_weak_diagrams';sys.dont_write_bytecode=True;checks=[];start=time.monotonic()
def gate(n,e,t=0.):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
initial_names=['results.json','extra_results.json','normalization_results.json','MANIFEST.json','REVIEW.md','INITIAL_EVIDENCE_PIN.json','reviewed_initial_author_external_legs.py.txt','reviewed_initial_author_results.json.txt','reviewed_initial_author_CHECKPOINT.json.txt'];initial={n:sha(HERE/n)for n in initial_names}
def authorhash():return {str(f.relative_to(AUTHOR)):sha(f)for f in AUTHOR.rglob('*')if f.is_file()and'__pycache__'not in str(f)}
before=authorhash();strongpath=ROOT/'ope_nucleon_amplitudes/ope.py';strongsha=sha(strongpath);expectedstrong=json.loads((HERE/'results.json').read_text())['author_before']['ope_nucleon_amplitudes/ope.py'];gate('same independently reviewed strong source',int(strongsha!=expectedstrong))
oldpath=HERE/'reviewed_initial_author_external_legs.py.txt';newpath=AUTHOR/'external_legs.py'
gate('exact initial reviewed source',int(sha(oldpath)!='372ae513793b7b2c359cdf57ed0566faec2af259e33683d4f52c68d882541484'))
gate('exact commissioned repair source',int(sha(newpath)!='d8400295e4bc286ed5453c6da70d5f3d1cfc76e94f926b3eb7603e0ea55d7d14'))
gate('author preservation equals independent copy',int((AUTHOR/'pre_helper_repair/external_legs.py').read_bytes()!=oldpath.read_bytes()))
def load(name,path):
 loader=importlib.machinery.SourceFileLoader(name,str(path));spec=importlib.util.spec_from_loader(name,loader);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;loader.exec_module(m);return m
O=load('original_weak_frozen_recheck',oldpath);N=load('repaired_weak_frozen_recheck',newpath)
points=json.loads((HERE/'results.json').read_text())['points'];p=np.array(points[0]['p']);om=points[0]['omega'];Q=np.array(points[0]['Q'])
# Reproduce exact original failures, don't merely assume old behavior.
oldnan=O.vector_defect_direct(p,om,Q,gv=float('nan'));gate('old NaNgv failure reproduced',int(np.all(np.isfinite(oldnan))))
oldraising=O.vector_defect_direct(p,om,Q,raising=float('nan'));oldtrue=O.vector_defect_direct(p,om,Q,raising=True);gate('old NaNraising wrong-branch failure reproduced',int(not np.array_equal(oldraising,oldtrue)))
for key,vals in [('gv',[float('nan'),float('inf'),False,1+0j,'1',None]),('raising',[float('nan'),1,0,'False',None])]:
 for val in vals:
  try:N.vector_defect_direct(p,om,Q,**{key:val});bad=1
  except ValueError:bad=0
  gate('repaired helper rejects '+key+'='+repr(val),bad)
# Same valid physics before and after repair, including raising/lowering.
for ix,row in enumerate(points):
 pp=np.array(row['p']);qq=np.array(row['Q']);ww=row['omega']
 for raising in[False,True,np.bool_(False)]:
  for comp in['V0','Vx','Vy','Vz','Ax','Ay','Az']:
   a=O.amplitude(pp,ww,qq,comp,gv=.83,ga=1.17,raising=raising);b=N.amplitude(pp,ww,qq,comp,gv=.83,ga=1.17,raising=raising);gate('valid finite amplitude BITWISE '+str((ix,str(raising),comp)),int(not np.array_equal(a,b)))
  a=O.vector_defect_direct(pp,ww,qq,gv=np.float64(.83),raising=raising);b=N.vector_defect_direct(pp,ww,qq,gv=np.float64(.83),raising=raising);gate('valid finite helper BITWISE '+str((ix,str(raising))),int(not np.array_equal(a,b)))
  lhs=ww*N.amplitude(pp,ww,qq,'V0',gv=.83,raising=raising)-sum(qq[j]*N.amplitude(pp,ww,qq,'V'+s,gv=.83,raising=raising)for j,s in enumerate('xyz'))
  rhs=b-N.ope.EXCHANGE@N.vector_defect_direct(pp[[0,1,3,2]],ww,qq,gv=.83,raising=raising);gate('repaired physical finiteQ Ward defect '+str((ix,str(raising))),np.max(abs(lhs-rhs))/np.max(abs(rhs)),3e-12)
# Structural diff confirms physical equations unchanged after removing only guards.
def functions(path):return {n.name:n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
a=functions(oldpath);b=functions(newpath)
for name in['validate','sigma','amplitude','channel','emission_point']:gate('unchanged exact function AST '+name,int(ast.dump(a[name],include_attributes=False)!=ast.dump(b[name],include_attributes=False)))
for name in['rho','direct','vector_defect_direct']:
 old=a[name];new=b[name]
 def stripped(body):
  out=[]
  for n in body:
   if isinstance(n,ast.Assign)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Name)and n.value.func.id=='_real_coupling':continue
   if isinstance(n,ast.If)and('math.isfinite'in ast.unparse(n.test)or('raising'in ast.unparse(n.test)and'isinstance'in ast.unparse(n.test))):continue
   out.append(ast.dump(n,include_attributes=False))
  return out
 gate('physical numerical body unchanged '+name,int(stripped(old.body)!=stripped(new.body)))
gate('initial review and two failures unchanged',int(initial!={n:sha(HERE/n)for n in initial_names}))
gate('strong source untouched during recheck',int(sha(strongpath)!=strongsha))
gate('author repaired sources/raw untouched during recheck',int(before!=authorhash()))
result=dict(checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],initial_evidence_before=initial,initial_evidence_after={n:sha(HERE/n)for n in initial_names},author_before=before,author_after=authorhash(),seconds=time.monotonic()-start,scope='helper-admission repair only; finiteQ interaction-current defect remains nonzero')
(HERE/'repair_recheck_results.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k]for k in['passed','failed','seconds']},indent=2))
