"""Locked selected fit data collection; no scientific execution on import."""
from pathlib import Path
import importlib.util,sys,json,hashlib,time,math,numbers
import numpy as np
P=Path(__file__).resolve().parent;R=P.parent;sys.dont_write_bytecode=True
LOCK_SHA='5b4ee8835a938b499fab0eba8095b04f18ef59b61dba957ee62220acaf7f225d'
def load(n,p):
 s=importlib.util.spec_from_file_location(n,R/p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
EC=load('compact_capture_EC_physical','hot_electron_capture_moments/electron_capture.py');PC=load('compact_capture_PC_physical','review_pc_moments/reference.py');S=load('compact_capture_static_allowed','weak_rates/static_capture.py')
PINS={'hot_electron_capture_moments/electron_capture.py':'b9954da0a2324376c19e4277013436b135d1432f9cb9904b03761be30bd1b3d1','review_pc_moments/reference.py':'2ca34b181865df529ee0b0d599faa932760190cfc25f97d71d513db6290ef133'}
Q=EC.MN-EC.MP
def verify():
 if hashlib.sha256((P/'REVISED_LOCK2.json').read_bytes()).hexdigest()!=LOCK_SHA:raise RuntimeError('Original locked experiment changed')
 for name,h in json.loads((P/'SOURCE_PIN.json').read_text()).items():
  if hashlib.sha256((R/name).read_bytes()).hexdigest()!=h:raise RuntimeError('Frozen source changed')
def basis(T,n,eta):return [1.,T,T*T,eta,eta*eta,T*eta,(n/1e-6)*T**-1.5]
def physical(state,refined):
 b,st=EC.matched(**state)
 n,y=state['nB'],state['Ye'];roots=dict(n=st['neutron']['density']/(n*(1-y))-1,p=st['proton']['density']/(n*y)-1,e=st['electron']['net_density']/(n*y)-1)
 if any(abs(x)>1e-10 for x in roots.values()):raise ArithmeticError('Forward actual EOS roots failed')
 ec=EC.integrate(b,proton_order=28 if refined else 20,electron_order=48 if refined else 32,initial_order=12 if refined else 8,outgoing_order=12 if refined else 8,phi_order=12 if refined else 8,tail=60. if refined else 50.,scenarios=[dict(name='empty',mode='empty')])
 pc=PC.integral(vars(b),order_n=32 if refined else 28,order_e=52 if refined else 44,order_angle=12 if refined else 10,tail=60.)
 ee=ec['scenarios']['empty']['net'];pp=np.asarray(pc['sums'])[0,0]
 return dict(state=state,bath=vars(b),EOS=st,root_residuals=roots,values=[ee['number'],ee['neutrino_energy'],float(pp[0]),float(pp[1])],raw=dict(EC=ec,PC=pc),density_residuals=dict(EC=ec['proton_density']/(state['nB']*state['Ye'])-1,PC=pc['neutron_density']/(state['nB']*(1-state['Ye']))-1))
def collect(group,index):
 verify()
 if group not in ['training','heldout']or type(index)is not int:raise ValueError('Declared group and literal index')
 lock=json.loads((P/'REVISED_LOCK2.json').read_text())
 if not 0<=index<len(lock[group]):raise ValueError('Selected index')
 if not json.loads((P/'rank_admission.json').read_text())['qualified']:raise RuntimeError('Blocked revised rank/admission')
 path=P/f'{group}_{index}.json'
 if path.exists():raise RuntimeError('Immutable completed state')
 state=lock[group][index]
 bb,_=EC.matched(**state)
 if abs(bb.mu_e/bb.T)>3:raise ValueError('Actual EOS outside retained physical admission')
 start=time.monotonic();a=physical(state,False);b=physical(state,True)
 checks=[]
 for k,(x,y)in enumerate(zip(a['values'],b['values'])):checks.append(dict(name='physical refinement output'+str(k),error=x/y-1,gate=3e-6,passed=bool(abs(x/y-1)<=3e-6)))
 for label,d in [('coarse',a),('refined',b)]:
  for ch,e in d['density_residuals'].items():checks.append(dict(name=label+' actualtargetdensity '+ch,error=e,gate=3e-8,passed=bool(abs(e)<=3e-8)))
 T,n,Y=state['T'],state['nB'],state['Ye'];mu=b['bath']['mu_e'];base=[]
 for ch,target in [('electron_capture',n*Y),('positron_capture',n*(1-Y))]:base.extend((target*S.target_moments(T,mu,ch,q=Q)).tolist())
 out=dict(group=group,index=index,state=state,coarse=a,refined=b,allowed=base,log_correction=np.log(np.asarray(b['values'])/base).tolist(),basis=basis(T,n,mu/T),checks=checks,qualified=all(c['passed']for c in checks),seconds=time.monotonic()-start,sourcepins=PINS)
 verify();path.write_text(json.dumps(out,indent=2)+'\n');return out
if __name__=='__main__':print(collect(sys.argv[1],int(sys.argv[2]))['qualified'])
