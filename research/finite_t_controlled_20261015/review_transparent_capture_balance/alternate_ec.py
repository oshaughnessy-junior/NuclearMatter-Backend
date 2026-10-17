"""Fresh alternate EC invariant-s/boost-axis/Dirac-trace phase at kinetic root."""
from pathlib import Path
import json,importlib.util,sys,math,time
P=Path(__file__).resolve().parent;ROOT=P.parent;sys.dont_write_bytecode=True
s=importlib.util.spec_from_file_location('review_balance_own_probe',P/'probe.py');R=importlib.util.module_from_spec(s);sys.modules[s.name]=R;s.loader.exec_module(R)
s=importlib.util.spec_from_file_location('review_balance_frozen_alternate_EC',ROOT/'review_hot_electron_capture/probe.py');I=importlib.util.module_from_spec(s);sys.modules[s.name]=I;s.loader.exec_module(I)
# Only pure integrate is called; frozen review main/driver never runs.
summary=json.loads((P/'summary.json').read_text());root=summary['kinetic'];original=json.loads((P/'raw.json').read_text());rows=[];checks=[];start=time.monotonic()
def ck(n,e,g):checks.append(dict(name=n,error=float(e),gate=g,passed=bool(math.isfinite(e)and abs(e)<=g)))
def save():
 (P/'alternate_ec.json').write_text(json.dumps(dict(rows=rows,checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],seconds=time.monotonic()-start),indent=2)+'\n')
for y,order,angle,phi,tail in [(root,24,10,8,70.),(root,32,12,10,70.),(.58,24,10,8,70.)]:
 b=R.eos(y);bath=I.C.Bath(**b);v=I.integrate(bath,order,angle,phi,tail);row=dict(Ye=y,source=v);rows.append(row)
 ck(f'alternate density {y} {order}',v['density']/(1e-7*y)-1,3e-8)
 ref=summary['balanced']['channels']['EC']if y==root else next(r for r in original['rows']if r['Ye']==.58)['channels']['EC']
 for k,value in zip(I.C.NAMES,v['scenarios']['empty']['net']):ck(f'fresh alternate-vs-scalar {y} {order} {k}',value/ref[k]-1,3e-6)
 ck(f'alternate fourmomentum {y} {order}',v['maxfour'],3e-10)
 save();print(y,order,v['seconds'],flush=True)
for k,a,b in zip(I.C.NAMES,rows[0]['source']['scenarios']['empty']['net'],rows[1]['source']['scenarios']['empty']['net']):ck('alternate sameYe 24to32 '+k,a/b-1,3e-6)
# Replace ONLY electron-capture moment in the independently assembled root ledger.
root_rates=summary['balanced']['channels'];fine=rows[1]['source']['scenarios']['empty']['net'];net=root_rates['beta']['number']+root_rates['PC']['number']-fine[0]
ck('mixed independent trace root absolute dYe/s',net/1e-7,3e-6)
# Two source controls at both fresh bracket states, own EOS rerun in every call.
for y in [.54,.58]:
 a=next(r for r in original['rows']if r['Ye']==y);b=R.source(y,True)
 for c in b['channels']:
  for k,v in b['channels'][c].items():ck(f'fresh bracket two rules {y} {c} {k}',a['channels'][c][k]/v-1,3e-6)
ck('frozen bytes unchanged',0 if R.before==R.snap()else 1,0)
save();print('DONE',len(checks),'failed',sum(not c['passed']for c in checks),'mixedrootdYe',net/1e-7,'seconds',time.monotonic()-start)
