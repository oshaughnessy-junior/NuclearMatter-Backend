"""Separate material/radiation entropy, EOS pair bookkeeping and final guards."""
from pathlib import Path
import json,math
import hot_moments as M
import hot_spectrum as S
H=Path(__file__).parent;d=json.loads((H/'results.json').read_text());r=d['rows'][-1]['result'];b=M.K.Bath(**r['bath']);checks=[];rows=[]
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and abs(error)<=tol)))
for label,s in r['scenarios'].items():
 if s['scenario']['mode']!='FD':continue
 R=s['net']['number'];Q=s['net']['neutrino_energy'];mu=s['scenario']['mu_nu'];scale=s['forward']['number']['value']+s['backward']['number']['value']
 sm=(-Q+(b.mu_n-b.mu_p-b.mu_e)*R)/b.T;sr=(Q-mu*R)/b.T
 gate('independent material+radiation entropy '+label,(sm+sr-s['entropy_production'])/scale,3e-12)
 gate('charge source '+label,s['number_sources']['proton']-s['number_sources']['electron_net'],0.)
 rows.append(dict(label=label,matter_entropy=sm,radiation_entropy=sr,total_entropy=sm+sr,dYe_dt=R/1e-7,dYp_dt=R/1e-7,material_total_energy_change=-Q))
meta=d['rows'][0]['matched_state'];e=meta['electron']
gate('pair neutrality',((e['particle_density']-e['antiparticle_density'])/meta['proton']['density'])-1,3e-10)
try:M.K.spectral(b,.42);rejected=False
except ValueError:rejected=True
gate('old cold spectrum still rejected at T2',int(not rejected),0.)
for name,fn in [('mixed radiation T',lambda:M.integrate(b,scenarios=[dict(name='bad',mode='FD',mu_nu=0,temperature=3)])),('changed selected spectral state',lambda:S.spectral(M.K.Bath(2,b.mu_n+1e-9,b.mu_p,b.mu_e),.42)),('changed selected spectral E',lambda:S.spectral(b,.43))]:
 try:fn();bad=True
 except ValueError:bad=False
 gate('reject '+name,int(bad),0)
old=json.loads((H/'controls.json').read_text());oldref=[z['new_hot_reference']for z in old['rows']if'new_hot_reference'in z][-1]
new=S.spectral(b,.42,angle_order=16,lepton_order=20,energy_order=24,widths=13.,tail=70.)
for key in ['j','chi']:gate('final exact-state guard replay '+key,new[key]['value']/oldref[key]['value']-1,0.)
rows.append(dict(final_guarded_spectrum=new))
(H/'bookkeeping.json').write_text(json.dumps(dict(checks=checks,rows=rows,all_pass=all(c['passed']for c in checks)),indent=2)+'\n')
print(json.dumps(dict(checks=len(checks),failed=[c for c in checks if not c['passed']])))
