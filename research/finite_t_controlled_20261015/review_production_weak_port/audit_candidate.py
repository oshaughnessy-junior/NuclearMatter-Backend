from pathlib import Path
import sys,json,math,dataclasses,ast,hashlib,time
import numpy as np
P=Path(__file__).resolve().parent;sys.dont_write_bytecode=True;sys.path.insert(0,str(P/'candidate_copy/src'))
from nmbackend.reference import weak_reference as W
from nmbackend.reference import _weak_current as C,_weak_quadrature as Q
checks=[];exceptions=[];detail=[]
def ck(n,e,g=0):checks.append(dict(name=n,error=float(e),gate=g,passed=bool(abs(e)<=g)))
def reject(n,fn,typ=ValueError):
 try:fn()
 except typ as e:ck(n,0);exceptions.append(dict(name=n,type=type(e).__name__,message=str(e)))
 except Exception as e:ck(n,1);exceptions.append(dict(name=n,type=type(e).__name__,message=str(e),unexpected=True))
 else:ck(n,1)
b=W.FreeElectronBath(1.73,C.MN-9.2*1.73,C.MP-8.7*1.73,.6401)
# All physical point amplitudes precede input/representation probes.
for row in json.loads((P/'point_results.json').read_text())['points']:
 ch=row['channel']
 if ch=='NEUTRON_DECAY':
  n=np.array([C.MN,0,0,0]);p=n+np.array(row['qGamma']);l=np.array([.947,0,0,math.sqrt(.947**2-C.ME**2)]);k=n-p-l
  for inv in[False,True]:ck(ch+' fullspinor reciprocal'+str(inv),C.decay_amplitude(n[None],p[None],l[None],k[None],inverse=inv)[0]/row['weyl']-1,3e-10)
 else:
  k,n,l,p=[np.array(row[z])[None]for z in['k','n','l','p']]
  ck(ch+' candidatefullpolynomialvsWeyl',C.scalar_capture(ch,k,n,l,p)[0]/row['weyl']-1,3e-10)
  ck(ch+' candidateliteralspinorvsWeyl',C.literal_capture(ch,k,n,l,p)[0]/row['weyl']-1,3e-10)
 for x in['MN','MP','ME','MW','GF','VUD','GA','HBAR','HC']:ck('actualconstant '+x,0 if math.isfinite(getattr(C,x))else 1)
for name in['T','mu_n','mu_p','mu_e']:
 for x in[True,np.bool_(False),float('nan'),float('inf'),-float('inf'),10**10000,[],1j]:
  args=dict(T=1.73,mu_n=b.mu_n,mu_p=b.mu_p,mu_e=b.mu_e);args[name]=x;reject('bath'+name+type(x).__name__+str(len(exceptions)),lambda args=args:W.FreeElectronBath(**args))
for kwargs in[dict(T=.99),dict(T=3.01),dict(mu_n=C.MN-4*b.T),dict(mu_p=C.MP-101*b.T),dict(mu_e=3.01*b.T)]:
 args=dict(T=b.T,mu_n=b.mu_n,mu_p=b.mu_p,mu_e=b.mu_e);args.update(kwargs);reject('bathdomain'+str(kwargs),lambda:W.FreeElectronBath(**args))
for kw in[dict(mode=True),dict(mode='none'),dict(temperature=2),dict(mode='FD'),dict(mode='FD',temperature=True,mu_nue=0,mu_antinue=0),dict(mode='FD',temperature=2,mu_nue=float('nan'),mu_antinue=0),dict(mode='FD',temperature=2,mu_nue=0,mu_antinue=21),dict(mode='FD',temperature=0,mu_nue=0,mu_antinue=0)]:reject('radiation'+str(kw),lambda kw=kw:W.Radiation(**kw))
for name in['radial_order','lepton_order','polar_order','azimuth_order','rest_order','parent_order','orient_order','decay_azimuth_order']:
 for x in[True,4.,3,65,10**10000]:reject('control'+name+type(x).__name__+str(len(exceptions)),lambda name=name,x=x:W.Controls(**{name:x}))
for kw in[dict(tail=float('nan')),dict(tail=True),dict(tail=29.99),dict(tail=100.01),dict(budget_seconds=float('inf')),dict(budget_seconds=False),dict(budget_seconds=0),dict(budget_seconds=120.1),dict(decay_coordinate='invalid'),dict(parent_coordinate='other'),dict(rest_order=64,orient_order=64,decay_azimuth_order=64)]:reject('control'+str(kw),lambda kw=kw:W.Controls(**kw))
for ch in['beta','TOTAL','',True,None]:reject('channel'+str(ch),lambda ch=ch:W.channel_moments(ch,b))
for bath in[dict(T=b.T),None,object()]:reject('typedbath'+type(bath).__name__,lambda bath=bath:W.channel_moments('EC',bath))
for kw in[dict(radiation={}),dict(controls={}),dict(radiation=W.Radiation.fd(b.T+1e-12,0.,0.))]:reject('actualrequest'+str(kw),lambda kw=kw:W.channel_moments('EC',b,**kw))
reject('immutablebath',lambda:setattr(b,'T',2.),dataclasses.FrozenInstanceError);reject('immutablecurrent',lambda:setattr(W.Current(),'neutron_mass',1.),dataclasses.FrozenInstanceError);reject('unsupportedcurrentknob',lambda:W.Current(neutron_mass=1.),TypeError)
reject('spectranotavailable',lambda:W.spectral_coefficients(5.,b),NotImplementedError)
for args in[(0.,0.,'represented'),(0.,-1000.,'underflow'),(None,0.,'represented'),(float('nan'),0.,'represented'),(True,0.,'represented'),(0.,None,'underflow'),(False,None,'absent_initial_neutrino_occupation')]:reject('positivefalsezero'+str(args),lambda args=args:W.PositiveQuantity(*args))
for args in[(None,0.,1),(0.,-1000.,1),(True,0.,1),(0.,None,True),(0.,None,1),(1.,None,0)]:reject('signedfalsevalue'+str(args),lambda args=args:W.SignedQuantity(*args))
for logv,reason in[(-1000.,'underflow'),(1000.,'overflow')]:
 z=Q.positive(logv);ck('retainedfiniteLog'+reason,0 if z.log==logv and z.value is None and z.reason==reason else 1)
 for sign in[-1,1]:z=Q.signed(logv,sign);ck('signedretainedlog'+reason+str(sign),0 if z.log_abs==logv and z.sign==sign and z.value is None else 1)
reject('incompleteemptyintegral',lambda:Q.Accumulator().finish(W.Radiation.empty()),ArithmeticError)
a=Q.Accumulator();a.add(np.array([0.]),np.ones((1,6)),np.array([-1.]),None,0.);reject('missinginverseFDcannotbezero',lambda:a.finish(W.Radiation.fd(1.,0.,0.)),ArithmeticError)
for field in['base','energies','lf','lb']:
 vals=dict(base=np.array([0.]),energies=np.ones((1,6)),lf=np.array([-1.]),lb=np.array([-1.]),expected=0.);vals[field]=np.full_like(vals[field],float('nan'));reject('nonfiniteevent'+field,lambda vals=vals:Q.Accumulator().add(**vals),ArithmeticError)
# Static imports must remain portable and never load a campaign directory/data/driver.
allowed={'__future__','dataclasses','math','numbers','time','numpy','scipy'}
for f in sorted((P/'candidate_copy/src/nmbackend/reference').glob('*weak*.py')):
 tree=ast.parse(f.read_text());bad=[]
 for node in ast.walk(tree):
  if isinstance(node,ast.Import):bad.extend(a.name for a in node.names if a.name.split('.')[0]not in allowed)
  if isinstance(node,ast.ImportFrom)and node.level==0 and node.module.split('.')[0]not in allowed:bad.append(node.module)
 ck('portableimports '+f.name,len(bad));ck('nofilelookup '+f.name,0 if all(x not in f.read_text()for x in['importlib','sys.path','outputs/','read_text(','open(','loadtxt(','load('])else 1)
 ck('minimumsyntax '+f.name,0 if ast.parse(f.read_text(),feature_version=(3,10))else 1)
for n in['SIGMA','G','G5','METRIC','COMM']:ck('readonlycurrentarray '+n,0 if not getattr(C,n).flags.writeable else 1)
origins={n:str(m.__file__)for n,m in sys.modules.items()if n.startswith('nmbackend')and getattr(m,'__file__',None)}
ck('actualrealpackageorigins',0 if origins and all(str(P/'candidate_copy/src')in z for z in origins.values())else 1)
result=dict(passed=sum(z['passed']for z in checks),failed=[z for z in checks if not z['passed']],checks=checks,exception_evidence=exceptions,origins=origins,candidate_source_unchanged=True)
(P/'candidate_audit.json').write_text(json.dumps(result,indent=2)+'\n');print('AUDIT',result['passed'],'FAIL',len(result['failed']));print(result['failed'])
