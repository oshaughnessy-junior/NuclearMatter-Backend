"""Independent composition/ledger and vacuum four-momentum threshold probes."""
from pathlib import Path
import sys,json,math,hashlib,time
import numpy as np
from scipy.optimize import brentq,minimize_scalar
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True;sys.path[:0]=[str(ROOT/'nr_baryon_eos'),str(ROOT/'thermodynamics')]
import free_nucleons as B
import lepton_reference as L
paths=list((ROOT/'muon_inventory').glob('*'));hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in paths if p.is_file()};start=time.monotonic();checks=[];raw={}
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and error<=tol)))
# New model assembly: baryon charge is sum of independent electron/muon net fractions.
def state(T,nB,Ye,Ymu):
 yp=Ye+Ymu;n=B.state_density(T,nB*(1-yp),B.Q);p=B.state_density(T,nB*yp,0.)
 e=L.state_density(T,nB*Ye,L.ELECTRON_MASS);m=L.state_density(T,nB*Ymu,L.MUON_MASS)
 u=(n.energy_density+p.energy_density+e.energy_density+m.energy_density)/nB
 Cv=(n.cv_fixed_density+p.cv_fixed_density+e.cv_fixed_net_density+m.cv_fixed_net_density)/nB
 an=n.mu-T*n.dmu_dT_fixed_density;ap=p.mu-T*p.dmu_dT_fixed_density
 ae=e.chemical_potential-T*e.dmu_dtemperature_fixed_net_density;am=m.chemical_potential-T*m.dmu_dtemperature_fixed_net_density
 return dict(u=u,derivatives=[Cv,ap-an+ae,ap-an+am],species=[n,p,e,m])
raw['thermodynamics']=[]
for T,nB,Ye,Ymu in [(9.,.02,.08,.015),(17.,.04,.045,.025)]:
 s=state(T,nB,Ye,Ymu);n,p,e,m=s['species']
 gate(f'charge {T}',abs((e.net_density+m.net_density)/p.density-1),2e-10)
 for h in [3e-5,1.5e-5]:
  v=[T,Ye,Ymu]
  for k in range(3):
   plus=v.copy();minus=v.copy();plus[k]+=h;minus[k]-=h
   fd=(state(plus[0],nB,plus[1],plus[2])['u']-state(minus[0],nB,minus[1],minus[2])['u'])/(2*h)
   gate(f'energy derivative {T} coordinate{k} h{h}',abs(fd/s['derivatives'][k]-1),2e-7)
 ze=e.chemical_potential+p.mu-n.mu;zm=m.chemical_potential+p.mu-n.mu
 gate(f'trapped muon decay affinity {T}',abs(m.chemical_potential-e.chemical_potential-zm+ze),3e-13)
 raw['thermodynamics'].append(dict(T=T,nB=nB,Ye=Ye,Ymu=Ymu,Yp=Ye+Ymu,u=s['u'],derivatives=s['derivatives'],mu_e=e.chemical_potential,mu_mu=m.chemical_potential,zeta_e=ze,zeta_mu=zm))
# Independently reconstruct the unique displayed event stoichiometry.
Q={'n':(1,0,0,0),'p':(1,1,0,0),'e-':(0,-1,1,0),'e+':(0,1,-1,0),'mu-':(0,-1,0,1),'mu+':(0,1,0,-1),'ve':(0,0,1,0),'ae':(0,0,-1,0),'vm':(0,0,0,1),'am':(0,0,0,-1)}
rx=[]
for l,v,a in [('e','ve','ae'),('mu','vm','am')]:rx += [([l+'-','p'],['n',v]),([l+'+','n'],['p',a]),(['n'],['p',l+'-',a]),(['p'],['n',l+'+',v])]
rx += [(['mu-'],['e-','ae','vm']),(['mu+'],['e+','ve','am']),(['ve','mu-'],['vm','e-']),(['ae','mu+'],['am','e+']),(['am','mu-'],['ae','e-']),(['vm','mu+'],['ve','e+']),(['e-','mu+'],['ve','am']),(['e+','mu-'],['ae','vm'])]
raw['reactions']=[]
for incoming,outgoing in rx:
 residual=np.sum([Q[x]for x in outgoing],axis=0)-np.sum([Q[x]for x in incoming],axis=0)
 gate('+'.join(incoming)+'->'+'+'.join(outgoing),max(abs(residual)),0.)
 raw['reactions'].append(dict(incoming=incoming,outgoing=outgoing,conservation_residual=residual.tolist()))
# Numerical threshold from minimum final lab energy at fixed total momentum E.
mn,mp=939.56542052,938.27208816;mm,me=L.MUON_MASS,L.ELECTRON_MASS
raw['thresholds']=[]
for name,target,final_baryon in [('numu+n',mn,mp),('anumu+p',mp,mn)]:
 def surplus(E):
  sol=minimize_scalar(lambda p:math.hypot(final_baryon,p)+math.hypot(mm,E-p),bounds=(0.,E),method='bounded',options={'xatol':1e-11})
  return sol.fun-target-E
 E=brentq(surplus,1.,200.,xtol=1e-11);formula=((final_baryon+mm)**2-target**2)/(2*target)
 gate(name+' independent minimum final-energy threshold',abs(E/formula-1),1e-10)
 gate(name+' below threshold forbidden',max(0.,-surplus(E-1e-3)),0.)
 raw['thresholds'].append(dict(process=name,threshold=E,formula=formula,static=final_baryon+mm-target))
E=brentq(lambda E:math.hypot(mm,E)-me-E,100.,1e5,xtol=1e-9)
gate('inverse muon decay lab energy closure',abs(E/((mm*mm-me*me)/(2*me))-1),2e-10)
raw['inverse_muon_decay_threshold_MeV']=E
raw['neutron_muon_decay_energy_excess']=mn-mp-mm
gate('neutron vacuum muon decay forbidden',max(0.,mn-mp-mm),0.)
end={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in paths if p.is_file()}
r=dict(passed=all(c['passed']for c in checks)and hashes==end,checks=checks,raw=raw,runtime_seconds=time.monotonic()-start,source_sha256=hashes,end_source_sha256=end,scope='Research contract review only: independent EOS composition derivatives, stoichiometry and vacuum support. No new weak-rate model or domain certificate.')
(HERE/'results.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(passed=r['passed'],checks=len(checks),runtime=r['runtime_seconds'],failed=[c for c in checks if not c['passed']]),indent=2))
