"""Bounded total-event proper-time, FD blocking, LTE and entropy ladder."""
from pathlib import Path
import json,math,time
import numpy as np
from scipy.integrate import quad
from scipy.special import expit
import moments as M
P=Path(__file__).resolve().parent;checks=[]
def ck(name,e,t):checks.append(dict(name=name,error=abs(float(e)),limit=t,passed=bool(abs(e)<=t)))
rows=[];start=time.monotonic()
for T in(.03,.1):
 bath,mat=M.K.matched_npe(T,1e-12,.3);aff=bath.mu_n-bath.mu_p-bath.mu_e
 scenarios=[dict(name='unblocked',mode='unblocked'),dict(name='empty',mode='empty')]+[dict(name=name,mode='FD',mu_nu=aff+offset*T)for name,offset in [('LTE',0),('emission',-1),('absorption',1)]]
 runs=[]
 for order,rad,ori,phi,coord in[(8,8,4,4,'pair'),(16,12,6,8,'pair'),(24,16,8,12,'pair'),(24,16,8,12,'electron')]:
  result=M.integrate(bath,rest_order=order,parent_order=rad,orient_order=ori,phi_order=phi,coordinate=coord,scenarios=scenarios);runs.append(result)
  (P/'partial_results.json').write_text(json.dumps(dict(rows=rows,current=dict(T=T,material=mat,runs=runs)),indent=2));print(T,order,coord,result['seconds'],flush=True)
 # Independent direct momentum integral for density and proper-time average.
 pmax=math.sqrt((M.K.MN+70*T)**2-M.K.MN**2)
 den=quad(lambda p:p*p*expit((bath.mu_n-math.hypot(M.K.MN,p))/T),0,pmax,epsabs=1e-16,epsrel=3e-11)[0]/(math.pi**2*M.HC**3)
 proper=quad(lambda p:p*p*M.K.MN/math.hypot(M.K.MN,p)*expit((bath.mu_n-math.hypot(M.K.MN,p))/T),0,pmax,epsabs=1e-16,epsrel=3e-11)[0]/(math.pi**2*M.HC**3)
 reference=M.B.pair_integral(32)
 for r in runs[-2:]:
  ck('FD_parent_density',r['parent_density']/mat['neutron']['density']-1,3e-10);ck('independent_parent_density',r['parent_density']/den-1,3e-10)
  u=r['scenarios']['unblocked']['forward'];ck('unblocked_proper_lifetime',u['number']['value']/(proper*reference['rate'])-1,3e-7)
  for key,ref in[('neutrino_energy',reference['energy_neutrino']),('electron_energy',reference['energy_electron']),('proton_energy',reference['energy_proton'])]:ck('unblocked_Doppler_cancellation_'+key,u[key]['value']/(den*ref)-1,3e-7)
  for name,s in r['scenarios'].items():
   ck('signed_energy_ledger_'+name,s['energy_ledger_scaled'],3e-9)
   ck('electron_lepton_ledger_'+name,s['matter_electron_lepton_change']+s['radiation_electron_lepton_change'],0.)
   if s['scenario']['mode']=='FD':
    ck('separate_FD_products_chemical_KMS',s['max_chemical_log_residual'],3e-9)
    flux=s['forward']['number']['value']+s['backward']['number']['value'];ck('FD_entropy_identity',(s['entropy_production']-s['net']['number']*s['common_T_affinity']/T)/flux,3e-9)
    assert s['entropy_production']>=0
    if name=='LTE':ck('LTE_net_number',s['net']['number']/flux,3e-9)
    else:assert s['net']['number']*s['common_T_affinity']>0
 for scenario in scenarios:
  name=scenario['name']
  for direction in('forward','backward'):
   for moment in M.NAMES:
    a=runs[-2]['scenarios'][name][direction][moment]['value'];b=runs[-1]['scenarios'][name][direction][moment]['value']
    if a and b:ck('independent_rest_parameterization_'+name+'_'+direction+'_'+moment,a/b-1,3e-6)
    a=runs[-3]['scenarios'][name][direction][moment]['value'];b=runs[-2]['scenarios'][name][direction][moment]['value']
    if a and b:ck('fine_numerical_refinement_'+name+'_'+direction+'_'+moment,a/b-1,3e-6)
 rows.append(dict(T=T,material=mat,runs=runs,direct_momentum_density=den,direct_proper_density=proper,reference_rest=reference))
out=dict(rows=rows,checks=checks,passed=sum(c['passed']for c in checks),failed=sum(not c['passed']for c in checks),seconds=time.monotonic()-start,scope='Event totals only; original tensor integrated over all Enu NOT evaluated')
(P/'results.json').write_text(json.dumps(out,indent=2));print(out['passed'],out['failed'],out['seconds'])
if out['failed']:raise SystemExit(1)
