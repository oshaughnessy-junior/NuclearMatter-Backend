from pathlib import Path
import sys,importlib.util,json,math,hashlib,numpy as np
P=Path(__file__).resolve().parent;sys.dont_write_bytecode=True
s=importlib.util.spec_from_file_location('optional_moment_API_controls',P/'approximation.py');A=importlib.util.module_from_spec(s);sys.modules[s.name]=A;s.loader.exec_module(A);checks=[]
def ck(n,v):checks.append(dict(name=n,passed=bool(v)))
for state in [(0.9,1e-7,.3),(3.1,1e-7,.3),(2.,9e-8,.3),(2.,1.1e-6,.3),(2.,1e-7,.19),(2.,1e-7,.51),(True,1e-7,.3),(2.,True,.3),(2.,1e-7,False),(float('nan'),1e-7,.3),(2.,float('inf'),.3),(2.,1e-7,float('nan')),(1.,1e-6,.5)]:
 try:A.approximate_moments(*state);rejected=False
 except ValueError:rejected=True
 ck('reject invalid or excluded state '+str(state),rejected)
for r in json.loads((P/'HELDOUT_ASSESSMENT.json').read_text())['rows']:
 v=A.approximate_moments(**r['state']);ck('API positive heldout'+str(r['index']),all(x>0 and math.isfinite(x)for x in v['moments'].values()));ck('API exact heldout arithmetic'+str(r['index']),list(v['moments'].values())==r['predicted']);ck('API no net source/root'+str(r['index']),set(v['moments'])=={'R_EC','Q_EC','R_PC','Q_PC'})
for group,n in [('training',11),('heldout',8)]:
 for i in range(n):
  r=json.loads((P/f'{group}_{i}.json').read_text());ck('selected sourcequalified '+str((group,i)),r['qualified']is True)
  for profile in ['coarse','refined']:ck('actual EOSroot '+str((group,i,profile)),max(abs(v)for v in r[profile]['root_residuals'].values())<=1e-10)
fit=json.loads((P/'COEFFICIENT_FREEZE.json').read_text())
for f,h in fit['training_files'].items():ck('training byteidentity '+f,hashlib.sha256((P/f).read_bytes()).hexdigest()==h)
old=json.loads((P/'LOCK.json').read_text());new=json.loads((P/'REVISED_LOCK2.json').read_text());ck('same7basis',old['basis']==new['basis']);ck('same8heldouts',old['heldout']==new['heldout']);ck('sameheldoutgate',old['heldout_gate']==new['heldout_gate']);ck('samephysicalgate',old['source_refinement_gate']==new['source_refinement_gate']);ck('removeONLYoutofadmissioncorner',[x for x in old['training']if x not in new['training']]==[dict(T=1.,nB=1e-6,Ye=.5)]and len(new['training'])==11)
ck('originalpreflightblockpreserved',json.loads((P/'preflight.json').read_text())['all_admitted']is False);ck('rankbeforedata',json.loads((P/'rank_admission.json').read_text())['rate_data_files_before_rank']==[]);ck('coefficientsbeforeheldout',fit['heldout_files_before_freeze']==[])
A.M.verify();out=dict(checks=checks,passed=sum(x['passed']for x in checks),failed=[x for x in checks if not x['passed']]);(P/'api_checks.json').write_text(json.dumps(out,indent=2)+'\n');print(out['passed'],out['failed'])
