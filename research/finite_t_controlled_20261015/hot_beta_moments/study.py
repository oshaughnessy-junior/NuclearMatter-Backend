"""Selected hot physical npe neutron beta moment pilot and refinement."""
from pathlib import Path
import json,math,time
import numpy as np
from scipy.integrate import quad
from scipy.special import expit
import hot_moments as M
H=Path(__file__).parent;checks=[];rows=[]
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and abs(error)<=tol)))
def save(stage):
 (H/'results.json').write_text(json.dumps(dict(stage=stage,checks=checks,rows=rows,all_pass=all(c['passed']for c in checks)),indent=2)+'\n')
b,metadata=M.K.matched_npe(2.,1e-7,.3);eq=b.mu_n-b.mu_p-b.mu_e
scenarios=[dict(name='unblocked',mode='unblocked'),dict(name='empty',mode='empty'),dict(name='LTE',mode='FD',mu_nu=eq),dict(name='emission_dominated',mode='FD',mu_nu=eq-.8*b.T),dict(name='inverse_dominated',mode='FD',mu_nu=eq+.8*b.T)]
rows.append(dict(matched_state=metadata,bath=vars(b),scenarios=scenarios))
n=quad(lambda p:p*p/(math.pi**2*M.HC**3)*expit((b.mu_n-math.hypot(M.K.MN,p))/b.T),0,900,epsabs=1e-21,epsrel=2e-12)[0]
proper=quad(lambda p:p*p/(math.pi**2*M.HC**3)*M.K.MN/math.hypot(M.K.MN,p)*expit((b.mu_n-math.hypot(M.K.MN,p))/b.T),0,900,epsabs=1e-21,epsrel=2e-12)[0]
rest=M.rest_events(40)
gate('independent neutron density forward',n/7e-8-1,3e-10)
rows.append(dict(independent_n=n,independent_nproper=proper,mean_m_over_E=proper/n,rest={k:v for k,v in rest.items()if not isinstance(v,np.ndarray)}));save('forward density')
for controls in [dict(rest_order=16,parent_order=12,orient_order=6,phi_order=8),dict(rest_order=24,parent_order=20,orient_order=10,phi_order=12),dict(rest_order=32,parent_order=28,orient_order=14,phi_order=20)]:
 r=M.integrate(b,scenarios=scenarios,**controls);rows.append(dict(label='Laguerre',result=r));save('hot event '+str(controls['rest_order']))
 gate('actual neutron density '+str(controls['rest_order']),r['parent_density']/n-1,3e-10)
 u=r['scenarios']['unblocked'];gate('true hot proper lifetime '+str(controls['rest_order']),u['forward']['number']['value']/(proper*rest['rate'])-1,3e-6)
 for key in ['neutrino_energy','electron_energy','proton_energy']:
  gate('hot Doppler '+str(controls['rest_order'])+' '+key,u['forward'][key]['value']/(n*rest[key])-1,3e-6)
 gate('hot parent energy '+str(controls['rest_order']),u['forward']['neutron_energy']['value']/(n*M.K.MN*rest['rate'])-1,3e-6)
 for label,s in r['scenarios'].items():
  scale=s['forward']['number']['value']+(s['backward']['number']['value'] or 0.)
  gate('physical energy ledger '+str(controls['rest_order'])+' '+label,s['energy_ledger_scaled'],3e-10)
  gate('direct material loss '+str(controls['rest_order'])+' '+label,s['direct_energy_ledger_scaled'],3e-10)
  gate('Le '+str(controls['rest_order'])+' '+label,(s['matter_electron_lepton_change']+s['radiation_electron_lepton_change'])/scale,2e-14)
  if s['scenario']['mode']=='FD':
   gate('local affinity '+str(controls['rest_order'])+' '+label,s['max_chemical_log_residual'],2e-11)
   gate('entropy positive '+str(controls['rest_order'])+' '+label,min(0,s['entropy_production']),0)
   target=s['net']['number']*s['common_T_affinity']/b.T
   gate('entropy affinity '+str(controls['rest_order'])+' '+label,(s['entropy_production']-target)/scale,3e-12)
   if label=='LTE':gate('actual LTE net '+str(controls['rest_order']),s['net']['number']/scale,3e-12)
   else:gate('source orientation '+str(controls['rest_order'])+' '+label,int((s['net']['number']>0)!=(label=='emission_dominated')),0)
runs=[z['result']for z in rows if z.get('label')=='Laguerre']
for i,r in enumerate(runs[:-1]):
 for label,s in r['scenarios'].items():
  for branch in ['forward','backward']:
   for key in M.NAMES:
    a=s[branch][key]['value'];z=runs[-1]['scenarios'][label][branch][key]['value']
    if z:gate('refinement'+str(i)+' '+label+' '+branch+' '+key,a/z-1,3e-6)
save('complete');print(json.dumps(dict(checks=len(checks),failed=[c for c in checks if not c['passed']])))
