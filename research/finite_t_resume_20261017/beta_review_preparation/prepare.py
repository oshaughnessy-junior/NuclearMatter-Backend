from pathlib import Path
from decimal import Decimal,localcontext
import json,hashlib,datetime
D=Path(__file__).resolve().parent; O=D.parents[1]/'nmb-finite-t-20261015'
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
paths=['hot_beta_moments/conditional.py','hot_beta_moments/hot_moments.py','hot_beta_moments/results.json','hot_beta_moments/hot_spectrum.py','relativistic_crossed_contract/beta.py','relativistic_crossed_contract/check_contract.py','relativistic_capture_oracle/events.py','relativistic_crossed_thermal/kernel.py','spectral_band_total_closure/REPORT.md','spectral_band_total_closure/ANALYSIS.json','spectral_band_total_closure/LOCK.json','spectral_band_total_closure/SOURCE_PIN.json','review_spectral_band_total_closure/REVIEW.md']
pins={str(O/p):h(O/p) for p in paths}
with localcontext() as c:
 c.prec=60;M=Decimal('939.56542052');mp=Decimal('938.27208816');me=Decimal('.51099895069');T=Decimal('2');q=(M*M-(mp+me)**2)/(2*M)
 rows=[]
 for estr in ('1.2','1.27'):
  E=Decimal(estr);d=M*q/E;A=(d*d+M*M)/(2*d);p=(M*M-d*d)/(2*d)
  assert A>M and p>0 and abs(A-p-d)<Decimal('1e-50')
  rows.append(dict(E_MeV=estr,parent_Emin_MeV=str(A),parent_pmin_MeV=str(p),parent_kinetic_min_MeV=str(A-M),D_MeV=str(d),maximum_lab_energy_at_threshold=str(q*(A+p)/M),target_tail_logoccupancy_scale=str((Decimal('919.8287578100662')-A)/T)))
 anchors={'precision':60,'rest_endpoint_MeV':str(q),'meaning':'analytic support and occupation scale ONLY, not j/chi','cases':rows}
(D/'ANALYTIC_SUPPORT.json').write_text(json.dumps(anchors,indent=2)+'\n')
lock={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'candidate_blind':True,'candidate_directories_read':[],'fresh_physics_calls':0,'quota_guard_utc':'2026-10-17 19:34:01 UTC','weekly_percent':93,'account':'82f4cfd9-9e20-47cf-9997-761328d89f60','limit':'codex','window_minutes':10080,'reset':1792652325,'heldout_E_MeV':1.27,'original_anchor_E_MeV':1.2,'future_max_fresh_integrals':4,'future_profile_plan':['rest-Dalitz conditional independent E1.2 lower','rest-Dalitz conditional independent E1.2 higher','rest-Dalitz conditional independent E1.27 higher','candidate E1.27 higher'],'gates':{'separate_positive_j_chi_relative':3e-6,'shell_relative':2e-11,'four_scaled':2e-12,'unsquared_delta_scaled':2e-12,'posterior_KMS_log':2e-9},'controls_initial_plan':{'lower':{'parent':16,'rest_nu':16,'electron':16,'phi':8},'higher':{'parent':24,'rest_nu':24,'electron':24,'phi':12},'parent_radial':'threshold-shifted semi-infinite; explicit Jacobi/Laguerre endpoint weights to be specified before first output after static feasibility audit'},'no_new_call_without_root_frozen_handoff':True,'source_sha256':pins}
(D/'LOCK.json').write_text(json.dumps(lock,indent=2)+'\n')
print(json.dumps(anchors,indent=2));print('LOCK',h(D/'LOCK.json'))
