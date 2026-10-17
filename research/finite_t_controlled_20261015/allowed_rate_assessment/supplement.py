from pathlib import Path
import importlib.util,sys,json,hashlib,math
P=Path(__file__).resolve().parent;sys.dont_write_bytecode=True
s=importlib.util.spec_from_file_location('allowed_supplement_no_driver',P/'assess.py');m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
d=json.loads((P/'results.json').read_text());checks=[];rest=[]
def ck(n,e,t):checks.append(dict(name=n,error=float(e),gate=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
for r in d['rows']:
 T,nb,y,mu=r['T'],r['nB'],r['detailed_kinetic_Ye'],r['actual_bath']['mu_e'];a=m.rates(T,mu,nb*(1-y),nb*y,96,70);b=m.rates(T,mu,nb*(1-y),nb*y,192,70);c=m.rates(T,mu,nb*(1-y),nb*y,96,90)
 for ch in m.CHANNELS:
  for k in range(2):ck(r['label']+' isolated order '+str((ch,k)),a[ch][k]/b[ch][k]-1,3e-10);ck(r['label']+' isolated tail '+str((ch,k)),a[ch][k]/c[ch][k]-1,3e-10)
 if r['label']=='original_anchor':src=json.loads((m.R/'transparent_capture_balance/results.json').read_text())['balanced']
 else:src=json.loads((m.R/f'transparent_balance_grid/state_{int(r["label"][-1])}.json').read_text())['balanced']
 beta=src['raw_sources']['beta'];width=beta['rest']['rate'];ratio=beta['parent_proper_density']/beta['parent_density']
 ck(r['label']+' physical vacuum rest width positive',0 if width>0 else 1,0);ck(r['label']+' proper-time ratio in(0,1)',0 if 0<ratio<1 else 1,0)
 rest.append(dict(label=r['label'],full_tree_rest_width=width,full_tree_rest_lifetime=1/width,allowed_over_full_rest_width_minus1=d['vacuum']['allowed_width_s1']/width-1,parent_proper_to_coordinate_density=ratio,derived_unblocked_full_rate=beta['parent_proper_density']*width,interpretation='Accepted immutable vacuum full-current rest integral and kinematic proper-density anchor. Warm blocked rows above are not unblocked vacuum rates. No new detailed phase integration.'))
review=m.R/'review_transparent_balance_grid/REVIEW.md';h=hashlib.sha256(review.read_bytes()).hexdigest();expected='0b2fe53e9a343a8a75f3cfbd38ea5afbb6bdc34895553202975c616deb9a988e';assert h==expected
out=dict(checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],vacuum_proper_anchors=rest,grid_independent_acceptance=dict(accepted=True,review=str(review.relative_to(m.R)),sha256=h,scope='Three selected free-npe fixed-T empty-radiation beta+PC+EC baths only'),source_hashes_unchanged=all(hashlib.sha256((m.R/f).read_bytes()).hexdigest()==v for f,v in d['pins'].items()))
(P/'supplement_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out['passed'],out['failed'],rest[0])
