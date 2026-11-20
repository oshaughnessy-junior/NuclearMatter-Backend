"""Independent fugacity/Bessel baryons and momentum leptons; new source roots."""
from pathlib import Path
import sys,math,json,hashlib,importlib.util,time
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import expit,kve
from scipy.optimize import brentq
P=Path(__file__).resolve().parent;ROOT=P.parent;A=ROOT/'transparent_balance_grid';sys.dont_write_bytecode=True
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
G=load('review_new_balance_frozen_functions',A/'grid.py');C=G.EC;HC=C.HC
checks=[];rows=[];start=time.monotonic();before={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in A.rglob('*')if p.is_file()};sourcepins=json.loads((A/'UPSTREAM.json').read_text())
def ck(n,e,t):checks.append(dict(name=n,error=float(e),gate=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
def density_eta(T,eta,m):
 r=np.arange(1,33);return m*m*T/math.pi**2/HC**3*sum((-1.)**(r+1)*np.exp(r*eta)*kve(2,r*m/T)/r)
def baryon_mu(T,n,m):return m+T*brentq(lambda eta:density_eta(T,eta,m)/n-1,-100.,-.001,xtol=2e-13,rtol=1e-14)
x,w=leggauss(80)
def electron_density(T,mu):
 result=0.
 for lo,hi in zip([0,1,4,10,30],[1,4,10,30,90]):
  p=T*(lo+(x+1)*(hi-lo)/2);E=np.hypot(p,C.ME);result+=sum(w*(hi-lo)*T/2*p*p*(expit((mu-E)/T)-expit((-mu-E)/T)))/(math.pi**2*HC**3)
 return float(result)
def eos(T,n,y):
 return dict(T=T,mu_n=baryon_mu(T,n*(1-y),C.MN),mu_p=baryon_mu(T,n*y,C.MP),mu_e=brentq(lambda mu:electron_density(T,mu)/(n*y)-1,0.,100.,xtol=2e-13,rtol=1e-14))
def source(T,n,y,fine=False):
 if time.monotonic()-start>105:raise TimeoutError('review source batch105s before next call')
 b=eos(T,n,y);bath=C.Bath(**b)
 ec=C.integrate(bath,proton_order=30 if fine else 24,electron_order=50 if fine else 40,initial_order=12 if fine else 10,outgoing_order=12 if fine else 10,phi_order=12 if fine else 10,tail=70. if fine else 55.,scenarios=[dict(name='empty',mode='empty')])
 pc=G.PC.integral(b,order_n=34 if fine else 30,order_e=54 if fine else 48,order_angle=12 if fine else 10,tail=70. if fine else 55.)
 bb=G.BETA.K.Bath(**b);beta=G.BETA.integrate(bb,rest_order=26 if fine else 20,parent_order=30 if fine else 26,orient_order=10 if fine else 8,phi_order=16 if fine else 12,coordinate='electron',parent_coordinate='direct_p',tail=90. if fine else 75.,scenarios=[dict(name='empty',mode='empty')])
 channels={'EC':ec['scenarios']['empty']['net'],'PC':dict(zip(G.PC.NAMES,map(float,np.asarray(pc['sums'])[0,0]))),'beta':beta['scenarios']['empty']['net']};rates={k:v['number']for k,v in channels.items()};R=rates['beta']+rates['PC']-rates['EC'];Q=sum(v['neutrino_energy']for v in channels.values());gross=sum(rates.values());direct=sum(v['material_energy_loss']for v in channels.values());material=channels['EC']['neutron_energy']-channels['beta']['neutron_energy']-channels['PC']['neutron_energy']+channels['beta']['proton_energy']+channels['PC']['proton_energy']-channels['EC']['proton_energy']+channels['beta']['electron_energy']-channels['PC']['positron_energy']-channels['EC']['electron_energy']
 row=dict(T=T,nB=n,Ye=y,fine=fine,bath=b,channels=channels,R=R,Q=Q,gross=gross,energy_ledger=(material+Q)/Q,direct_ledger=(Q-direct)/Q,density=dict(EC=ec['proton_density']/(n*y)-1,PC=pc['neutron_density']/(n*(1-y))-1,beta=beta['parent_density']/(n*(1-y))-1))
 rows.append(row)
 for k,v in row['density'].items():ck(f'fresh source density {T}/{n}/{y}/{fine}/{k}',v,3e-8)
 ck(f'independent species energy {T}/{y}/{fine}',row['energy_ledger'],3e-10);ck(f'direct energy {T}/{y}/{fine}',row['direct_ledger'],3e-10);ck(f'positive escaping energy {T}/{y}/{fine}',0 if Q>0 else 1,0)
 (P/'partial.json').write_text(json.dumps(dict(rows=rows,checks=checks),indent=2)+'\n');return row

def main():
 chemical=[];authors=[]
 ck('pinned source',0 if hashlib.sha256((A/'grid.py').read_bytes()).hexdigest()=='44402616f9b966ace87981534659477e01a2a8cb9e54127c9bc84e6b8ea4c9cc'else 1,0)
 ck('pinned manifest',0 if before['transparent_balance_grid/MANIFEST.json']=='783d87df1812da87628f729ec08135bee3df1415e3981298c1c6444d6317383b'else 1,0)
 G.require_acceptance()
 for i,(T,n)in enumerate(G.STATES):
  a=json.loads((A/f'state_{i}.json').read_text());authors.append(a);y=a['kinetic_Ye'];own=eos(T,n,y);b,_=G.material(T,n,y)
  for k in ['mu_n','mu_p','mu_e']:ck(f'independent EOS {i}/{k}',own[k]-getattr(b,k),3e-10)
  yc=brentq(lambda y:(lambda b:b['mu_n']-b['mu_p']-b['mu_e'])(eos(T,n,y)),.05,.9,xtol=1e-12);chemical.append(yc);ck(f'independent chemical root {i}',yc-a['chemical_Ye'],3e-10)
  r=source(T,n,y);f=source(T,n,y,True)
  ck(f'fresh root gross normalized {i}',f['R']/f['gross'],1e-7)
  for ch in f['channels']:
   for k,v in f['channels'][ch].items():
    ck(f'independent controls {i}/{ch}/{k}',r['channels'][ch][k]/v-1,3e-6);ck(f'fresh vs saved author {i}/{ch}/{k}',v/a['balanced']['channels'][ch][k]-1,3e-6)
 # One new kinetic root from fresh sources, not saved interpolation.
 T,n=G.STATES[0];y=authors[0]['kinetic_Ye'];lo=source(T,n,y-.001);hi=source(T,n,y+.001);ck('new left bracket sign',0 if lo['R']>0 else 1,0);ck('new right bracket sign',0 if hi['R']<0 else 1,0)
 ya=lo['Ye'];yb=hi['Ye'];fa=lo['R'];fb=hi['R']
 for j in range(3):
  yc=yb-fb*(yb-ya)/(fb-fa);r=source(T,n,yc);ya,fa,yb,fb=yb,fb,yc,r['R']
  if abs(fb/r['gross'])<2e-9:break
 ck('independent fresh kinetic root gross',r['R']/r['gross'],1e-7);ck('independent kinetic root value',r['Ye']-y,3e-8)
 after={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in A.rglob('*')if p.is_file()};ck('author bytes unchanged',int(before!=after),0)
 for p,h in sourcepins.items():ck('upstream frozen '+p,0 if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h else 1,0)
 out=dict(checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],rows=rows,chemical_roots=chemical,independent_kinetic_root=r,seconds=time.monotonic()-start,before=before,after=after);(P/'results.json').write_text(json.dumps(out,indent=2)+'\n');print(out['passed'],out['failed'],'seconds',out['seconds'])
if __name__=='__main__':main()
