"""Bounded actual FD source controls, independent current and moment gates."""
from pathlib import Path
import json,math,time,hashlib
import numpy as np
from scipy.integrate import quad
from scipy.special import expit
import electron_capture as C
HERE=Path(__file__).resolve().parent;checks=[];rules=[];start=time.monotonic()
def ck(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(np.isfinite(error)and abs(error)<=tol)))
def dump(name,data):(HERE/name).write_text(json.dumps(data,indent=2)+'\n')
bath,state=C.matched();dump('MATCHED_STATE.json',state)
# Forward densities before event or sensitivity comparisons; no MB replacement.
for species,mass,mu,wanted in [('neutron',C.MN,bath.mu_n,7e-8),('proton',C.MP,bath.mu_p,3e-8)]:
 own=quad(lambda p:p*p*expit((mu-math.hypot(p,mass))/bath.T),0,900,epsabs=1e-16,epsrel=2e-12)[0]/(math.pi**2*C.HC**3)
 ck('independent FD '+species+' density',own/wanted-1,3e-10)
minus=quad(lambda p:p*p*expit((bath.mu_e-math.hypot(p,C.ME))/bath.T),0,200,epsabs=1e-14,epsrel=2e-12)[0]/(math.pi**2*C.HC**3)
plus=quad(lambda p:p*p*expit((-bath.mu_e-math.hypot(p,C.ME))/bath.T),0,200,epsabs=1e-14,epsrel=2e-12)[0]/(math.pi**2*C.HC**3)
ck('independent FD pair net charge',(minus-plus)/3e-8-1,3e-10)
for name,value in [('particle_density',minus),('antiparticle_density',plus)]:ck('actual pair density '+name,value/state['electron'][name]-1,3e-10)
anchors=json.loads((HERE/'initial_amplitude_anchors.json').read_text())
for i,r in enumerate(anchors['rows']):
 ck('literal reciprocal vs scalar '+str(i),r['relative_scalar'],3e-12)
 ck('literal vs neutrino-capture adjoint '+str(i),r['relative_reciprocity'],3e-12)
# Actual full-event scalar/spinor total at same deliberately coarse nodes.
pilot=dict(proton_order=8,electron_order=12,initial_order=4,outgoing_order=4,phi_order=4,tail=40.)
a=C.integrate(bath,**pilot,amplitude='spinor');dump('literal_spinor_pilot.json',a)
b=C.integrate(bath,**pilot,amplitude='scalar');dump('scalar_pilot.json',b)
for scenario in a['scenarios']:
 for direction in ['forward','backward']:
  for key in C.NAMES:
   x=a['scenarios'][scenario][direction][key]['value'];y=b['scenarios'][scenario][direction][key]['value']
   if x:ck('same-event integrated amplitude '+str((scenario,direction,key)),y/x-1,3e-11)
# Save every completed rule before attempting a new one. Coarse failures retained.
for rule in [(12,20,6,40.),(20,32,8,50.),(28,48,12,60.)]:
 n,e,ang,tail=rule
 r=C.integrate(bath,proton_order=n,electron_order=e,initial_order=ang,outgoing_order=ang,phi_order=ang,tail=tail)
 rules.append(r);dump('partial_results.json',dict(rules=rules,checks=checks,elapsed=time.monotonic()-start))
 q=r['scenarios'];ck('proton phase density '+str(rule),r['proton_density']/3e-8-1,3e-8)
 ck('event threshold shell '+str(rule),r['max_threshold_shell_residual'],3e-12)
 ck('event fourmomentum '+str(rule),r['max_four_residual']/C.MN,3e-12)
 ck('event neutron mass '+str(rule),r['max_neutron_mass_residual'],3e-12)
 for name,s in q.items():
  ck('restinclusive E ledger '+str((rule,name)),s['energy_ledger_scaled'],3e-11)
  ck('direct restinclusive E ledger '+str((rule,name)),s['direct_energy_ledger_scaled'],3e-11)
  scale=s['forward']['number']['value']+(s['backward']['number']['value']or 0.)
  ck('net electron lepton ledger '+str((rule,name)),(s['matter_Le_change']+s['radiation_Le_change'])/scale,3e-13)
  ck('baryon ledger '+str((rule,name)),(s['number_sources']['n']+s['number_sources']['p'])/scale,3e-13)
  ck('charge ledger '+str((rule,name)),(s['number_sources']['p']-s['number_sources']['electron_net'])/scale,3e-13)
  if name!='empty':
   ck('direct occupation KMS '+str((rule,name)),s['max_event_kms_error'],3e-11)
   ck('entropy sign '+str((rule,name)),min(s['entropy'],0)/scale,1e-14)
  if name=='LTE':ck('LTE net/gross '+str(rule),s['net']['number']/scale,3e-12)
 if len(rules)>1:
  old,new=rules[-2:]
  for scenario in q:
   for direction in ['forward','backward']:
    for key in C.NAMES:
     x=old['scenarios'][scenario][direction][key]['value'];y=new['scenarios'][scenario][direction][key]['value']
     if y:ck('completed rule refinement '+str((len(rules),scenario,direction,key)),x/y-1,3e-6)
 dump('partial_results.json',dict(rules=rules,checks=checks,elapsed=time.monotonic()-start))
out=dict(checks=checks,passed=sum(x['passed']for x in checks),failed=[x for x in checks if not x['passed']],rules=rules,spinor_pilot=a,scalar_pilot=b,bath=vars(bath),state=state,seconds=time.monotonic()-start,source_sha256=hashlib.sha256((HERE/'electron_capture.py').read_bytes()).hexdigest(),qualification='Selected actualFD freeEC event source; inspect coarse failures before accepting fine rule. No crosschanneljoin/opacity/fullinventory.')
dump('results.json',out);print(json.dumps(dict(passed=out['passed'],failed=out['failed'],seconds=out['seconds'],fine_empty=rules[-1]['scenarios']['empty']['net'])))
