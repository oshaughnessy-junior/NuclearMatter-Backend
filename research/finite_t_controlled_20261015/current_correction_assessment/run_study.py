"""Serial fresh-process phase studies; baseline forward gate precedes F2 switch."""
from pathlib import Path
import sys,json,math,hashlib,subprocess,time
P=Path(__file__).resolve().parent;ROOT=P.parent;lock=json.loads((P/'LOCK.json').read_text());start=time.monotonic();checks=[];rows=[]
def ck(n,e,t=3e-6):checks.append(dict(name=n,error=float(e),gate=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
def save():
 (P/'partial.json').write_text(json.dumps(dict(checks=checks,rows=rows,seconds=time.monotonic()-start,status='INCOMPLETE_CHECKPOINT'),indent=2)+'\n')
if(P/'results.json').exists():raise RuntimeError('Immutable finished study')
for p,h in lock['pins'].items():
 if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:raise RuntimeError('Pinned source input changed')
control=json.loads((P/'current_controls.json').read_text())
if control['failed']:raise RuntimeError('Current forward algebra gates failed')
ref=json.loads((ROOT/'transparent_balance_grid/state_1.json').read_text())['balanced'];groups={}
for variant in ['baseline','F2zero']:
 values=[]
 for profile in [0,1]:
  path=P/f'{variant}_profile{profile}.json'
  if path.exists():raise RuntimeError('Completed child exists; no silent resume')
  r=subprocess.run([sys.executable,'-B',str(P/'worker.py'),variant,str(profile)],capture_output=True,text=True,timeout=115)
  (P/f'{variant}_profile{profile}.stdout').write_text(r.stdout);(P/f'{variant}_profile{profile}.stderr').write_text(r.stderr)
  if r.returncode:save();raise RuntimeError('Child failure retained '+variant+str(profile))
  v=json.loads(path.read_text());values.append(v);rows.append(v);save()
  for ch in ['EC','PC']:
   ck(f'{variant}{profile} actual target density {ch}',v['density_residuals'][ch],lock['gates']['FD_target_density'])
   for j,k in enumerate(['number','neutrino_energy']):
    ck(f'{variant}{profile} positive absolute {ch}/{k}',0 if v['values'][ch][j]>0 else 1,0)
    if variant=='baseline':ck(f'baseline forward actual {profile}/{ch}/{k}',v['values'][ch][j]/ref['channels'][ch][k]-1,lock['gates']['baseline_absolute_RQ'])
  n=lock['state']['nB'];y=lock['state']['Ye'];ck(f'{variant}{profile} actual EOSn',v['EOS']['neutron']['density']/(n*(1-y))-1,lock['gates']['EOS']);ck(f'{variant}{profile} actual EOSp',v['EOS']['proton']['density']/(n*y)-1,lock['gates']['EOS']);ck(f'{variant}{profile} actual charge',v['EOS']['electron']['net_density']/(n*y)-1,lock['gates']['EOS'])
  if any(not c['passed']for c in checks):save();raise ArithmeticError('Forward/numerical gate failed; no variant admission')
 for ch in ['EC','PC']:
  for j,k in enumerate(['number','neutrino_energy']):ck(f'{variant} independentprofile {ch}/{k}',values[0]['values'][ch][j]/values[1]['values'][ch][j]-1,lock['gates']['individual_profile_refinement'])
 if any(not c['passed']for c in checks):save();raise ArithmeticError('Profile gate failed; no retuning')
 groups[variant]=values
 # This entire baseline loop completes and passes BEFORE launching F2zero.
full=groups['baseline'][1];off=groups['F2zero'][1];comparison={};allowed=next(r for r in json.loads((ROOT/'allowed_rate_assessment/results.json').read_text())['rows']if r['label']=='new_state_1')
ck('same actual bath under current switch',0 if full['bath']==off['bath']else 1,0);ck('same actual EOS under current switch',0 if full['EOS']==off['EOS']else 1,0)
for ch in ['EC','PC']:
 comparison[ch]=[]
 for j,name in enumerate(['number','neutrino_energy']):
  baseline=full['values'][ch][j];switched=off['values'][ch][j];simple=allowed['channels'][ch]['allowed'][j]
  comparison[ch].append(dict(moment=name,baseline=baseline,F2zero=switched,F2_removal_fraction=switched/baseline-1,F2_addition_fraction=baseline/switched-1,allowed=simple,allowed_over_baseline_minus1=simple/baseline-1,allowed_over_F2zero_minus1=simple/switched-1))
for p,h in lock['pins'].items():ck('external frozen unchanged '+p,0 if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h else 1,0)
out=dict(status='AUTHOR_FROZEN_DIFFERENT_REVIEW_REQUIRED',checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],current_control_passed=control['passed'],comparison=comparison,rows=rows,seconds=time.monotonic()-start,combined_physical_calls=4,lock_sha256=hashlib.sha256((P/'LOCK.json').read_bytes()).hexdigest());(P/'results.json').write_text(json.dumps(out,indent=2)+'\n');print(out['passed'],out['failed'],'seconds',out['seconds']);print(comparison)
