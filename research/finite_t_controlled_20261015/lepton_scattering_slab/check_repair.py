"""Scoped guard regressions and bitwise-valid original physical replay."""
import sys,json,types,importlib.util,hashlib,copy,math,time,ast
from pathlib import Path
import numpy as np
from scipy.special import expit
P=Path(__file__).parent;ROOT=P.parent;sys.dont_write_bytecode=True;oldpath=P/'initial_input_failure_snapshot';checks=[];start=time.monotonic()
def ck(n,ok,details=None):checks.append(dict(name=n,passed=bool(ok),details=details))
def reject(n,fn):
 try:fn();ok=False
 except ValueError:ok=True
 ck(n,ok)
sp=importlib.util.spec_from_file_location('repaired_slab_guard_author',P/'slab.py');B=importlib.util.module_from_spec(sp);sys.modules[sp.name]=B;sp.loader.exec_module(B)
# Execute exact frozen owned source with its original path for upstream resolution.
original=(oldpath/'slab.py').read_text();Old=types.ModuleType('preserved_original_slab_guard');Old.__file__=str((P/'slab.py').resolve());sys.modules[Old.__name__]=Old;exec(compile(original,Old.__file__,'exec'),Old.__dict__)
r=json.loads((oldpath/'accepted_results.json').read_text());g=r['grids'][-1];E=np.array(g['E']);N=len(E);left,right=B.boundary(E);f=np.tile(expit((-.7-E)/2.),(2,1))
# Guard rejections must precede any attempted numerical ODE solve.
old_solver=B.solve_ivp
def unexpected(*a,**k):raise RuntimeError('ODE invoked before control guard')
B.solve_ivp=unexpected
for field in ['left_T','right_T','initial_T']:
 for val in [-3.,0.,float('nan'),float('inf'),-float('inf'),True]:reject('preODE positive thermal '+field+repr(val),lambda field=field,val=val:B.evolve(g,cells=2,duration=.003,**{field:val}))
for field in ['duration','crossing_time','rtol','max_step','rad_mu']:
 for val in [float('nan'),float('inf'),True]:reject('preODE finite control '+field+repr(val),lambda field=field,val=val:B.evolve(g,**{field:val}))
B.solve_ivp=old_solver
for collisions in[False,True]:
 for key in ['f','left','right']:
  for val in [float('nan'),float('inf'),-float('inf'),0.,1.,-.2]:
   a=f.copy();l=left.copy();rr=right.copy();target={'f':a,'left':l,'right':rr}[key];target.flat[0]=val
   reject('rhs invalid '+str((collisions,key,val)),lambda a=a,l=l,rr=rr,collisions=collisions:B.rhs(g,a,l,rr,collisions=collisions))
 for flag in [0,1,None,'yes']:reject('literalcollision flag '+repr(flag),lambda flag=flag:B.rhs(g,f,left,right,collisions=flag))
 for val in [float('nan'),float('inf'),True,0.,-1.]:reject('rhs crossing scale '+repr(val),lambda val=val:B.rhs(g,f,left,right,collisions=collisions,crossing_time=val))
for shape in [left[:-1],left[:,None],np.array(.3)]:reject('boundaryshape '+str(shape.shape),lambda shape=shape:B.rhs(g,f,shape,right,collisions=False))
for states in [f.ravel(),f[:0],f[:,:-1],f.astype(complex)]:reject('cellshape/type '+str((states.shape,str(states.dtype))),lambda states=states:B.rhs(g,states,left,right,collisions=False))
for key,val in [('E',float('nan')),('W',0.),('mu',float('nan')),('mu',1.),('A',float('nan'))]:
 gg=copy.deepcopy(g);arr=np.array(gg[key]);arr.flat[0 if key!='A' else 1]=val;gg[key]=arr.tolist();reject('invalid physical grid '+str((key,val)),lambda gg=gg:B.rhs(gg,f,left,right,collisions=False))
for Ebad in [[float('nan')],[0.],[-1.],np.array([1.+1j])]:reject('finite positive boundary E '+repr(Ebad),lambda Ebad=Ebad:B.boundary(Ebad))
# Valid finite calculations must remain bitwise identical, not merely within a relaxed gate.
for lT,rT,mu in [(3.,1.3,-.7),(2.,2.,.4)]:
 a=B.boundary(E,lT,rT,mu);b=Old.boundary(E,lT,rT,mu);ck('bitwise literal boundary '+str((lT,rT,mu)),all(np.array_equal(x,y)for x,y in zip(a,b)))
for coll in [False,True]:
 fnew=np.tile(expit((-.41-E)/1.85),(2,1));a,la=B.rhs(g,fnew,left,right,crossing_time=.047,collisions=coll);b,lb=Old.rhs(g,fnew,left,right,crossing_time=.047,collisions=coll);ck('bitwise physical rhs '+str(coll),np.array_equal(a,b)and np.array_equal(la,lb))
 a=B.collision(g,fnew[0]);b=Old.collision(g,fnew[0]);ck('bitwise paired collision '+str(coll),all(np.array_equal(a[k],b[k])for k in a))
args=dict(cells=2,duration=.013,crossing_time=.047,left_T=2.7,right_T=1.6,rad_mu=-.41,initial_T=1.85,max_step=.001)
a=B.evolve(g,**args);b=Old.evolve(g,**args);ck('bitwise full finite trajectory excluding runtime',{k:v for k,v in a.items()if k!='seconds'}=={k:v for k,v in b.items()if k!='seconds'})
# Compare unchanged arithmetic AST sections, guarding against an accidental physical edit.
trees=[]
for src in [original,(P/'slab.py').read_text()]:trees.append({n.name:n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef)})
for name in ['angular_edge','grid']:ck('identical full physical function AST '+name,ast.dump(trees[0][name])==ast.dump(trees[1][name]))
# Arithmetic tails are copied untouched after the original control/array setup.
for name,marker in [('collision','dn=np.zeros(N)'),('rhs','# Face flux'),('evolve','def fun(t,y):')]:
 def tail(src):
  body=src[src.index('def '+name+'('):];end=body.find('\ndef ',5);body=body if end<0 else body[:end];return body[body.index(marker):]
 ck('identical physical arithmetic source tail '+name,tail(original)==tail((P/'slab.py').read_text()))
# Original author/reviewer failures and pins remain byte-for-byte untouched.
pins=json.loads((oldpath/'SNAPSHOT_PINS.json').read_text());ck('preserved initialauthor snapshot',all(hashlib.sha256((oldpath/n).read_bytes()).hexdigest()==sha for n,sha in pins.items()))
out=dict(checks=checks,passed=sum(x['passed']for x in checks),failed=[x for x in checks if not x['passed']],valid_repaired=a,valid_original=b,initial_source_sha=hashlib.sha256(original.encode()).hexdigest(),repaired_source_sha=hashlib.sha256((P/'slab.py').read_bytes()).hexdigest(),seconds=time.monotonic()-start)
(P/'repair_results.json').write_text(json.dumps(out,indent=2));print(json.dumps(dict(passed=out['passed'],failed=out['failed'],seconds=out['seconds']),indent=2))
