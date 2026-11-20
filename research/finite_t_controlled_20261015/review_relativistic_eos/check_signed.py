from pathlib import Path
import sys,json,math,hashlib
import numpy as np
from scipy.integrate import quad
from scipy.special import expit
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'relativistic_free_eos'));import free_baryons as A
checks=[];rows=[]
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
for T,n,Ye,Ymu in [(4.1,.0011,-.027,.19),(2.9,.0017,.22,0.)]:
 m=A.matter_state(T,n,Ye,Ymu)
 for name,st,target in [('electron',m.electron,n*Ye),('muon',m.muon,n*Ymu)]:
  mass=st.mass;mu=st.chemical_potential;end=math.sqrt((max(mass,abs(mu))+60*T)**2-mass*mass)
  def fn(p):
   E=math.hypot(mass,p);return p*p*(expit((mu-E)/T)-expit((-mu-E)/T))/(math.pi**2*A.HBARC**3)
  value=quad(fn,0,end,epsabs=1e-16,epsrel=3e-11,limit=250)[0]
  gate(f'{T} {name} independent signed net root',(value-target)/max(n,abs(target)),3e-10)
  if target==0:gate(f'{T} {name} pair gas retained',float(st.particle_density<=0 or st.antiparticle_density<=0 or st.chemical_potential!=0),0.)
 gate(f'{T} two-lepton neutrality',(m.electron.net_density+m.muon.net_density-m.proton.density)/n,3e-10)
 gate(f'{T} independent affinities',m.zeta_e-m.zeta_mu-(m.electron.chemical_potential-m.muon.chemical_potential),5e-13)
 rows.append(dict(T=T,nB=n,Ye=Ye,Ymu=Ymu,zeta_e=m.zeta_e,zeta_mu=m.zeta_mu,mu_e=m.electron.chemical_potential,mu_mu=m.muon.chemical_potential))
Path(__file__).with_name('signed_results.json').write_text(json.dumps(dict(passed=all(c['passed']for c in checks),checks=checks,rows=rows),indent=2)+'\n');print(len(checks),all(c['passed']for c in checks))
