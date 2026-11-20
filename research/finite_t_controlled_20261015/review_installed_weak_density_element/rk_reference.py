"""Independent RK4 assembly; only accepted pure EOS and event functions imported."""
from pathlib import Path
import json,hashlib,sys,math,time,importlib.util
import numpy as np
from scipy.optimize import brentq
P=Path(__file__).resolve().parent;R=P.parent;A=R/'installed_weak_density_element';sys.dont_write_bytecode=True
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
frozen=json.loads((P/'AUTHOR_FROZEN.json').read_text());assert sha(A/'MANIFEST.json')==frozen['manifest'];manifest=json.loads((A/'MANIFEST.json').read_text());before={k:sha(A/k)for k in manifest['files']};assert before==manifest['files']
L=json.loads((P/'VALIDATION_LOCK.json').read_text());steps=int(sys.argv[1]);assert steps in L['steps'];start=time.monotonic();checks=[];records=[];calls=[]
def ck(name,error,gate):checks.append(dict(name=name,error=float(error),gate=gate,passed=bool(abs(error)<=gate)))
M=load('review_weak_accepted_binding',R/'installed_free_matter_binding/matter.py');W=load('review_weak_accepted_event_functions',R/'three_source_cooling_element/worker.py');W.pins();B=W.physical()
q=L['initial'];T0=q['T'];n0=q['nB'];Y0=q['Ye'];k=L['density']['k'];end=L['density']['duration'];a0=M.state(T0,n0,Y0);u0=a0.energy/n0;sigma0=a0.entropy/n0
prior=json.loads((R/'review_installed_free_matter_binding/results.json').read_text())['details']['forward_0']['oracle']
for name in ['energy','pressure','entropy','cv','e_n','e_Y']:ck('initial independent momentum '+name,getattr(a0,name)/prior[name]-1,3e-9)
# State vector u,Y,Esc_nue,Esc_anti,N_nue,N_anti,work,sigma_integral.
x=np.array([u0,Y0,0.,0.,0.,0.,0.,0.]);dt=end/steps

def event(a,profile):
 if time.monotonic()-start>105:raise TimeoutError('Independent reference budget, completed-call checkpoints retained')
 bath=B.EC.Bath(**a.weak_bath());cp=W.CONTROLS[profile];z=cp['EC']
 ec=B.EC.integrate(bath,proton_order=z[0],electron_order=z[1],initial_order=z[2],outgoing_order=z[2],phi_order=z[2],tail=z[3],scenarios=[dict(name='empty',mode='empty')],budget_seconds=90.)
 z=cp['PC'];pc=B.PC.integral(vars(bath),order_n=z[0],order_e=z[1],order_angle=z[2],tail=z[3],budget=90.)
 z=cp['beta'];be=B.BETA.integrate(B.BETA.K.Bath(**vars(bath)),rest_order=z[0],parent_order=z[1],orient_order=z[2],phi_order=z[3],coordinate=z[4],parent_coordinate=z[5],tail=z[6],scenarios=[dict(name='empty',mode='empty')],budget_seconds=90.)
 channels=dict(EC=ec['scenarios']['empty']['net'],PC=dict(zip(B.PC.NAMES,map(float,pc['sums'][0][0]))),beta=be['scenarios']['empty']['net'])
 rates={c:v['number']for c,v in channels.items()};Qe=channels['EC']['neutrino_energy'];Qa=channels['PC']['neutrino_energy']+channels['beta']['neutrino_energy'];Rnet=rates['beta']+rates['PC']-rates['EC']
 den=[ec['proton_density']/(a.nB*a.Ye)-1,pc['neutron_density']/(a.nB*(1-a.Ye))-1,be['parent_density']/(a.nB*(1-a.Ye))-1]
 ck(f'call{len(calls)}actualFDdensity',max(abs(z)for z in den),3e-8)
 for ch,v in channels.items():
  ck(f'call{len(calls)}{ch}directmaterialloss',(v['material_energy_loss']-v['neutrino_energy'])/v['neutrino_energy'],2e-9)
  if ch=='EC':ledger=v['neutron_energy']-v['proton_energy']-v['electron_energy']+v['neutrino_energy']
  elif ch=='PC':ledger=v['proton_energy']-v['neutron_energy']-v['positron_energy']+v['neutrino_energy']
  else:ledger=v['proton_energy']+v['electron_energy']-v['neutron_energy']+v['neutrino_energy']
  ck(f'call{len(calls)}{ch}speciesE',ledger/v['neutrino_energy'],2e-9)
 ret=dict(T=a.T,nB=a.nB,Y=a.Ye,profile=profile,channels=channels,R=Rnet,Qe=Qe,Qa=Qa,rates=rates,density_residuals=den);calls.append(ret)
 save();return ret

def state(t,x):
 n=n0*math.exp(k*t);Y=float(x[1]);u=float(x[0]);T=brentq(lambda T:M.state(T,n,Y,photons=True).specific_energy-u,1.5,2.5,xtol=1e-12,rtol=1e-14);a=M.state(T,n,Y)
 ck('energy inversion thermal scaled',(a.specific_energy-u)/(a.cv*T/n),2e-10);return a

def save():
 out=dict(steps=steps,checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],records=records,calls=calls,seconds=time.monotonic()-start)
 (P/f'RK4_{steps}.json').write_text(json.dumps(out,indent=2)+'\n')

def rhs(t,x):
 a=state(t,x);z=event(a,'base');n=a.nB;Q=z['Qe']+z['Qa'];rr=z['R'];work=a.pressure*k/n;sig=(-Q-(a.proton.chemical_potential+a.electron.chemical_potential-a.neutron.chemical_potential)*rr)/(n*a.T)
 dy=rr/n;du=work-Q/n
 # Independent chain-rule reconstruction back into analytic T equation.
 td1=(du-(a.e_n/n-a.energy/n**2)*k*n-a.e_Y/n*dy)/(a.cv/n)
 td2=(((a.energy+a.pressure)/n-a.e_n)*k*n-Q-a.e_Y*dy)/a.cv
 ck('instantaneous analytic temperature chain',(td1-td2)/max(abs(td1),abs(td2)),2e-9)
 return np.array([du,dy,z['Qe']/n,z['Qa']/n,z['rates']['EC']/n,(z['rates']['PC']+z['rates']['beta'])/n,work,sig])

for i in range(steps):
 t=i*dt;aa=rhs(t,x);bb=rhs(t+dt/2,x+dt*aa/2);cc=rhs(t+dt/2,x+dt*bb/2);dd=rhs(t+dt,x+dt*cc);x=x+dt*(aa+2*bb+2*cc+dd)/6;a=state(t+dt,x);records.append(dict(t=t+dt,state=a.asdict(),vector=list(map(float,x))))
 save()
a=state(end,x);grossE=x[2]+x[3]+abs(x[6]);grossN=x[4]+x[5]
ck('global material work escaping E',(x[0]-u0-x[6]+x[2]+x[3])/grossE,2e-8)
ck('global electronfamily Le',(x[1]-Y0+x[4]-x[5])/grossN,2e-8)
ck('global material entropy',(a.entropy/a.nB-sigma0-x[7])/abs(x[7]),1e-4)
ck('positive actual escapedE',0 if min(x[2],x[3],x[4],x[5])>0 else 1,0)
# Fine endpoint calls belong only to4step reference, avoiding duplicatecoverage.
if steps==4:
 b=event(a,'base');f=event(a,'fine')
 for ch in f['channels']:
  for name,v in f['channels'][ch].items():ck('fresh endpoint refine '+ch+name,b['channels'][ch][name]/v-1,3e-6)
assert {key:sha(A/key)for key in manifest['files']}==before;W.pins();ck('all author bytes unchanged',0,0);save();print('PASS',sum(c['passed']for c in checks),'FAIL',sum(not c['passed']for c in checks),'seconds',time.monotonic()-start)
