"""New-state direct-Dirac EC and independent Dalitz beta; no frozen driver execution."""
from pathlib import Path
import importlib.util,sys,json,math,hashlib,time
import numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parent;sys.dont_write_bytecode=True
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
R=load('grid_review_own_functions',P/'probe.py');I=load('grid_review_EC_dirac_functions',ROOT/'review_hot_electron_capture/probe.py');D=load('grid_review_beta_dalitz_functions',ROOT/'review_crossed_moments/reference.py');A=ROOT/'transparent_balance_grid';checks=[];rows=[];start=time.monotonic();before={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in A.rglob('*')if p.is_file()}
def ck(n,e,t=3e-6):checks.append(dict(name=n,error=float(e),gate=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
a=json.loads((A/'state_1.json').read_text());y=a['kinetic_Ye'];b=R.eos(3.,1e-7,y);bath=I.C.Bath(**b)
for order,angle,phi in [(24,10,8),(32,12,10)]:
 z=I.integrate(bath,order,angle,phi,70.);rows.append(dict(channel='EC',raw=z));ck('EC independent parent density '+str(order),z['density']/(1e-7*y)-1,3e-8)
 for k,v in zip(I.C.NAMES,z['scenarios']['empty']['net']):ck('EC newT3 directtrace '+str(order)+' '+k,v/a['balanced']['channels']['EC'][k]-1)
for k,x,yv in zip(I.C.NAMES,rows[0]['raw']['scenarios']['empty']['net'],rows[1]['raw']['scenarios']['empty']['net']):ck('EC alternate refinement '+k,x/yv-1)
a=json.loads((A/'state_0.json').read_text());b=R.eos(1.,1e-7,a['kinetic_Ye']);bath=type('OwnExplicitBath',(),b)();last=[]
for o,rad,ang,phi in [(20,10,8,12),(28,16,12,16)]:
 z=D.integrate(bath,bath.mu_n-bath.mu_p-bath.mu_e,order=o,radial=rad,orient=ang,phi=phi,budget=100);last.append(z);rows.append(dict(channel='beta',raw=z));ck('beta independent parent density '+str(o),z['density']/(1e-7*(1-a['kinetic_Ye']))-1,3e-8)
 for k,v in zip(D.NAMES,z['scenarios']['empty']['net']):ck('beta newT1 Dalitz '+str(o)+' '+k,v/a['balanced']['channels']['beta'][k]-1)
for k,x,yv in zip(D.NAMES,last[0]['scenarios']['empty']['net'],last[1]['scenarios']['empty']['net']):ck('beta alternate refinement '+k,x/yv-1)
after={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in A.rglob('*')if p.is_file()};ck('author bytes unchanged',int(before!=after),0)
out=dict(rows=rows,checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],seconds=time.monotonic()-start);(P/'alternate_results.json').write_text(json.dumps(out,indent=2,default=lambda x:x.tolist()if isinstance(x,np.ndarray)else x.item())+'\n');print(out['passed'],out['failed'],'seconds',out['seconds'])
