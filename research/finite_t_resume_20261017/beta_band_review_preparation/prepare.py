from pathlib import Path
import json,hashlib,datetime
import numpy as np
from numpy.polynomial.legendre import leggauss
D=Path(__file__).resolve().parent;R=D.parent;h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();B=R/'review_beta_support';M=json.loads((B/'MANIFEST.json').read_text());assert h(B/'CHECKPOINT.json')=='2cae22ff1c83537f94e272364e17d807723ed2d1855c47c53c8bd5eeb9859716'
assert all(h(B/f)==s for f,s in M['files_sha256'].items())
x,w=leggauss(24);en=1.1+(x+1)*.15;we=w*.15
nodes=[{'index':i,'E_MeV':float(a),'weight_MeV':float(b)}for i,(a,b)in enumerate(zip(en,we))]
lock={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'candidate_band_blind':True,'candidate_band_directories_read':[],'fresh_physics_calls':0,'interval_MeV':[1.1,1.4],'incident_rule':{'name':'Gauss-Legendre','order':24,'nodes':nodes},'baseline_inner':{'parent':24,'rest_nu':24,'electron':24,'phi':12,'parent_domain':'threshold-shifted semi-infinite generalizedLaguerre alpha1.5'},'fresh_call_max':28,'actual_calls_require_root_frozen_handoff':True,'inner_checks':[{'index':i,'name':name,**params}for i in(0,23)for name,params in[('parent_only',{'parent':32,'rest_nu':24,'electron':24,'phi':12}),('angular_coupled',{'parent':24,'rest_nu':32,'electron':32,'phi':16})]],'radiation':[{'name':'empty','mode':'empty'},{'name':'FD_muanti0','mode':'FD','T_MeV':2.,'mu_antinu_MeV':0.}],'gates':{'separate_positive_number_energy_branch_relative':3e-6,'inner_positive_j_chi_relative':3e-6,'structural_empty_inverse':'exact zero, no relative comparison','source_shell_relative':2e-11,'source_four_scaled':2e-12,'source_unsquared_delta_scaled':2e-12},'quota':{'fresh_utc':'2026-10-17 19:49:35 UTC','account':'82f4cfd9-9e20-47cf-9997-761328d89f60','limit':'codex','window_minutes':10080,'reset':1792652325,'used_percent':94,'stop_new':97,'checkpoint':99},'source_pins':{str(B/f):s for f,s in M['files_sha256'].items()},'source_review_checkpoint':h(B/'CHECKPOINT.json'),'source_review_manifest':h(B/'MANIFEST.json'),'same_current_and_parent_radial_method':True,'whole_energy_qualified':False,'original_12_failed_gates_remain_failed':True}
(D/'LOCK.json').write_text(json.dumps(lock,indent=2)+'\n')
assert all(n['weight_MeV']>0 for n in nodes);assert abs(sum(we)-.3)<1e-15
print('LOCK',h(D/'LOCK.json'));print('edge energies',en[0],en[-1]);print('sourcefiles',len(lock['source_pins']))
