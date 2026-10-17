"""Independent repaired-source recheck; all initial review evidence untouched."""
from pathlib import Path
import sys,types,ast,json,hashlib,copy
import numpy as np
from scipy.special import expit
HERE=Path(__file__).resolve().parent;AUTHOR=HERE.parent/'lepton_scattering_slab';sys.dont_write_bytecode=True
oldbytes=(HERE/'reviewed_initial_slab.py').read_bytes();newbytes=(AUTHOR/'slab.py').read_bytes()
expected='13c3bee74aa03803693416082271b5fad2dba5d6e8e1759e9eb2020be6694dfa'
assert hashlib.sha256(newbytes).hexdigest()==expected
initial=json.loads((HERE/'MANIFEST.json').read_text());pins={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()for name in initial['files']}
def load(name,source):
 m=types.ModuleType(name);m.__file__=str(AUTHOR/'slab.py');exec(compile(source,str(AUTHOR/'slab.py'),'exec'),m.__dict__);return m
old=load('initial_slab_recheck',oldbytes);new=load('repaired_slab_recheck',newbytes)
g=json.loads((HERE/'reviewed_initial_accepted_results.json').read_text())['grids'][-1]
E=np.array(g['E']);mu=np.array(g['mu']);f=np.array([expit((-.52-E)/(1.7+.23*mu)),expit((-.12-E)/(2.1-.17*mu))]);left=expit((-.52-E)/2.4);right=expit((-.52-E)/1.7);checks=[]
def ck(name,passed,**ev):checks.append(dict(name=name,passed=bool(passed),**ev))
oldcounter=[]
for name,fn in [('NaNleft',lambda:old.rhs(g,f,np.full_like(left,np.nan),right,collisions=False)),('NaNstate',lambda:old.rhs(g,np.full_like(f,np.nan),left,right,collisions=False)),('negative_temperature',lambda:old.evolve(g,cells=2,duration=.003,left_T=-3.,max_step=.001))]:
 r=fn();nonfinite=not all(np.all(np.isfinite(x))for x in r)if isinstance(r,tuple)else False
 ck('exact initial '+name+' failure reproduced',nonfinite if name!='negative_temperature'else isinstance(r,dict))
 oldcounter.append(dict(name=name,returned_nonfinite=nonfinite,completed_ODE=isinstance(r,dict)))

bad=[]
for label,badvalue in [('NaN',np.nan),('plusInf',np.inf),('minusInf',-np.inf),('zero',0.),('one',1.)]:
 for switch in [False,True]:
  for mode in ['state','left','right']:
   ff=f.copy();ll=left.copy();rr=right.copy()
   if mode=='state':ff[0,1]=badvalue
   elif mode=='left':ll[1]=badvalue
   else:rr[1]=badvalue
   bad.append((str((label,switch,mode)),lambda ff=ff,ll=ll,rr=rr,switch=switch:new.rhs(g,ff,ll,rr,collisions=switch)))
for name,kwargs in [('negative_left',{'left_T':-3.}),('negative_right',{'right_T':-1.}),('negative_initial',{'initial_T':-1.}),('NaNinitial',{'initial_T':np.nan}),('NaNmu',{'rad_mu':np.nan}),('NaNrtol',{'rtol':np.nan}),('negative_rtol',{'rtol':-1.}),('Infstep',{'max_step':np.inf}),('NaNduration',{'duration':np.nan}),('integer_flag',{'collisions':1})]:
 bad.append((name,lambda kwargs=kwargs:new.evolve(g,**kwargs)))
for name,value in [('complex',f.astype(complex)),('string',f.astype(str)),('bool',f>0),('shape',f.ravel()),('zero_cells',f[:0])]:
 bad.append(('state_'+name,lambda value=value:new.rhs(g,value,left,right,collisions=False)))
bad.extend([('left_wrongshape',lambda:new.rhs(g,f,left[None,:],right)),('right_complex',lambda:new.rhs(g,f,left,right.astype(complex))),('rhs_NaNcrossing',lambda:new.rhs(g,f,left,right,crossing_time=np.nan)),('rhs_integer_flag',lambda:new.rhs(g,f,left,right,collisions=1))])

# Invalid controls must fail before any numerical evolution begins.
solve=new.solve_ivp
def forbidden(*a,**k):raise RuntimeError('ODE entry reached on invalid input')
new.solve_ivp=forbidden
for name,fn in bad:
 try:fn();message='NOT REJECTED';ok=False
 except ValueError as e:message=str(e);ok=True
 except Exception as e:message=type(e).__name__+':'+str(e);ok=False
 ck('repaired rejects '+name,ok,message=message)
new.solve_ivp=solve

for name,key,value in [('NaNW','W',np.nan),('negativeW','W',-1.),('NaNE','E',np.nan),('zeroE','E',0.),('NaNmu','mu',np.nan),('negativeA','A',-1.)]:
 gg=copy.deepcopy(g)
 if key=='A':gg[key][0][1]=value
 else:gg[key][0]=value
 try:new.collision(gg,f[0]);ok=False;message='NOT REJECTED'
 except ValueError as e:ok=True;message=str(e)
 ck('repaired grid rejects '+name,ok,message=message)
for switch in [False,True]:
 a,b=old.rhs(g,f,left,right,crossing_time=.043,collisions=switch);aa,bb=new.rhs(g,f,left,right,crossing_time=.043,collisions=switch)
 ck('fresh valid RHS bitwise '+str(switch),np.array_equal(a,aa));ck('fresh valid face+thermostat ledgers bitwise '+str(switch),np.array_equal(b,bb))
for z in range(2):
 a=old.collision(g,f[z]);b=new.collision(g,f[z]);ck('fresh valid paired collision bitwise '+str(z),all(np.array_equal(a[k],b[k])for k in a))
a=old.boundary(E,2.4,1.7,-.52);b=new.boundary(E,2.4,1.7,-.52);ck('fresh valid boundary bitwise',all(np.array_equal(x,y)for x,y in zip(a,b)))
opts=dict(cells=2,duration=.011,crossing_time=.043,left_T=2.4,right_T=1.7,rad_mu=-.52,initial_T=1.93,max_step=.0007,rtol=3e-11)
a=old.evolve(g,**opts);b=new.evolve(g,**opts);ck('new complete valid trajectory bitwise',all(np.array_equal(a[k],b[k])for k in a if k!='seconds'))

def funcs(src):return {n.name:n for n in ast.parse(src).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
fa,fb=funcs(oldbytes),funcs(newbytes)
for name in ['angular_edge','grid']:
 ck(name+' complete AST unchanged',ast.dump(fa[name],include_attributes=False)==ast.dump(fb[name],include_attributes=False))
for name,marker in [('collision','dn'),('rhs','faces')]:
 def tail(fn):
  for i,node in enumerate(fn.body):
   if isinstance(node,ast.Assign)and any(isinstance(t,ast.Name)and t.id==marker for t in node.targets):return ast.dump(ast.Module(body=fn.body[i:],type_ignores=[]),include_attributes=False)
 ck(name+' physical arithmetic AST unchanged',tail(fa[name])==tail(fb[name]))
def ode_tail(fn):
 for i,node in enumerate(fn.body):
  if isinstance(node,ast.FunctionDef)and node.name=='fun':return ast.dump(ast.Module(body=fn.body[i:],type_ignores=[]),include_attributes=False)
ck('ODE and ledger arithmetic AST unchanged',ode_tail(fa['evolve'])==ode_tail(fb['evolve']))
ck('initial review evidence byte-identical',pins=={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()for name in pins})
ck('new author source still frozen',hashlib.sha256((AUTHOR/'slab.py').read_bytes()).hexdigest()==expected)
upstream=json.loads((HERE/'reviewed_initial_accepted_results.json').read_text())['pins']
ck('all four elementary physical sources match initial pinned hashes',all(hashlib.sha256((HERE.parent/name).read_bytes()).hexdigest()==h for name,h in upstream.items()))
out=dict(checks=checks,passed=sum(x['passed']for x in checks),failed=[x for x in checks if not x['passed']],oldcounterexamples=oldcounter,new_trajectory=b,initial_review_pins=pins,new_source_sha256=expected,scope='input contract repair accepted only if all gates pass; physicalfinitegrid/time qualification unchanged')
(HERE/'repair_recheck_results.json').write_text(json.dumps(out,indent=2,default=lambda x:x.item()if isinstance(x,np.generic)else x.tolist())+'\n');print(json.dumps({k:out[k]for k in ['passed','failed']}))
