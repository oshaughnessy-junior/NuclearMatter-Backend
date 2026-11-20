"""Replay all four independent blockers using only owned copies and synthetic wire mutations."""
from __future__ import annotations
import copy,json,subprocess,sys,shutil,time
from pathlib import Path
import client as C
P=C.P;HERE=Path(__file__).resolve().parent;start=time.monotonic();checks=[];artifacts=[]
def gate(n,ok,detail=None):checks.append(dict(name=n,passed=bool(ok),detail=detail))
def rejects(n,fn):
 try:fn();accepted=True
 except (ValueError,RuntimeError,TypeError):accepted=False
 gate(n,not accepted)
c=C.Client();raw=P.loads((HERE/'results.json').read_text());out=raw['real_calls'][0]['response'];req=raw['real_calls'][0]['request']
for name in sorted(P.REQUIRED_SOURCES):
 pins=copy.deepcopy(c._pins);del pins['sources'][name];rejects('reject missing mandatory pin '+name,lambda pins=pins:P.verify_pins(pins))
pins=copy.deepcopy(c._pins);pins['sources']['ghost.py']='0'*64;rejects('reject extra unexpected source pin',lambda:P.verify_pins(pins))
# Exact independent missing-PC+changed-source counterexample, with execution marker.
for missing in [True,False]:
 fixture=HERE/('repair_missing_pc_mirror'if missing else'repair_changed_pc_mirror')
 for name in c._pins['sources']:
  target=fixture/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(P.ROOT/name,target)
 pins=copy.deepcopy(c._pins)
 if missing:del pins['sources']['review_pc_moments/reference.py']
 path=fixture/'review_pc_moments/reference.py';path.write_text(path.read_text()+"\nfrom pathlib import Path as _SentinelPath\n_SentinelPath(__file__).with_name('EXECUTED').write_text('should never run')\n")
 (fixture/'isolated_microphysics/PINS.json').write_text(P.dumps(pins));r=copy.deepcopy(req);r['identity']=P.digest(pins)
 done=subprocess.run([sys.executable,'-I','-B',str(fixture/'isolated_microphysics/worker_entry.py')],input=P.dumps(r),text=True,capture_output=True,timeout=10,cwd=HERE);answer=P.loads(done.stdout)
 gate('actual changed PC rejected before import; missing='+str(missing),done.returncode==2 and answer['status']=='error'and not path.with_name('EXECUTED').exists(),answer);artifacts.append(dict(fixture=str(fixture),response=answer))
rejects('explicit spec source-path substitution rejects',lambda:P.verify_file(HERE/'repair_changed_pc_mirror/review_pc_moments/reference.py','review_pc_moments/reference.py',c._pins))
for branch in ['neutron_beta','positron_capture']:
 z=copy.deepcopy(out);other='positron_capture'if branch=='neutron_beta'else'neutron_beta'
 for key in ['number','energy']:
  z['sources']['channels'][branch][key]=P.represented(None,zero_reason='empty_incident_radiation');z['sources']['channels'][other][key]=copy.deepcopy(z['sources']['total'][key])
 rejects('reject false zero emission with combined totals transferred '+branch,lambda z=z:c.validate(z,req))
for val in [False,True,'0',None]:
 z=copy.deepcopy(out);z['sources']['inverse']['value']=val;rejects('reject inverse malformed numeric '+repr(val),lambda z=z:c.validate(z,req))
for field in ['value','log']:
 z=copy.deepcopy(out);z['sources']['channels']['neutron_beta']['number'][field]=False;rejects('reject boolean emission '+field,lambda z=z:c.validate(z,req))
z=copy.deepcopy(out);z['eos']['density_residual']=False;rejects('reject boolean forward residual',lambda:c.validate(z,req))
for v in [0,0.]:
 z=copy.deepcopy(out);z['sources']['inverse']['value']=v
 try:c.validate(z,req);ok=True
 except Exception:ok=False
 gate('retain numeric exact inverse zero '+repr(v),ok)
# Physical final report already contains bounded actual fresh initial/changed calls.
gate('all four repaired real calls present',len(raw['real_calls'])==4 and raw['passed']==112 and not raw['failed'])
result=dict(checks=checks,passed=sum(x['passed']for x in checks),failed=[x for x in checks if not x['passed']],mirrors=artifacts,pins=c._pins,seconds=time.monotonic()-start)
(HERE/'repair_results.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k]for k in ['passed','failed','seconds']},indent=2))
