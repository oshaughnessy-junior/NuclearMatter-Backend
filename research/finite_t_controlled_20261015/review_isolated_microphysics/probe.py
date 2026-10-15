"""Independent isolation/metadata adversaries; owned mirrors only."""
import sys,types,importlib.util,json,hashlib,copy,subprocess,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OWN=Path(__file__).parent;LEAF=ROOT/'isolated_microphysics';sys.dont_write_bytecode=True
checks=[]
def check(name,passed,details=None):checks.append(dict(name=name,passed=bool(passed),details=details))
def snapshot():
 dirs=['isolated_microphysics','transparent_antinu_element','hot_beta_moments','review_pc_moments','relativistic_crossed_contract','relativistic_crossed_thermal','relativistic_free_eos','thermodynamics','relativistic_tensor','relativistic_capture_oracle','muon_full_current','muon_vacuum']
 return{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for d in dirs for p in(ROOT/d).rglob('*')if p.is_file()and'__pycache__'not in str(p)}
before=snapshot();start=time.monotonic();sentinels={}
for n in ['study','protocol','hot_moments','kernel','beta','vacuum_cc','current','events','free_baryons','lepton_reference']:
 sentinels[n]=types.ModuleType(n);sys.modules[n]=sentinels[n]
spec=importlib.util.spec_from_file_location('independent_isolated_client_exact',LEAF/'client.py');C=importlib.util.module_from_spec(spec);sys.modules[spec.name]=C;spec.loader.exec_module(C)
c=C.Client();P=C.P;baselinepath=list(sys.path);baselinemods=dict(sys.modules)
rows=[]
for T,Y,profile in [(2.,.3,'base'),(1.9955690477112278,.3218092326059971,'fine'),(2.,.3,'fine')]:
 req=c.request(C.MaterialState(T,1e-7,Y),C.Controls(profile));out=c.evaluate(C.MaterialState(T,1e-7,Y),C.Controls(profile));rows.append(dict(request=req,response=out))
 check('fresh physical source positive '+str((T,Y,profile)),out['sources']['total']['number']['value']>0 and out['sources']['total']['energy']['value']>0)
 check('colliding parent modules remain identical '+str(profile),all(sys.modules[n]is m for n,m in sentinels.items()))
check('parent path unchanged',sys.path==baselinepath);check('no physical parent module changed',all(sys.modules.get(n)is m for n,m in baselinemods.items()));check('no new parent modules',set(sys.modules)==set(baselinemods))
# Concrete fresh heldout physical forward values from prior independent Heun endpoint.
end=rows[1]['response'];check('heldout number agrees prior independent forward',abs(end['sources']['total']['number']['value']/1.0642163330393762e-6-1)<2e-12)
check('heldout energy agrees prior independent forward',abs(end['sources']['total']['energy']['value']/1.1400114218759336e-5-1)<2e-12)
# A valid current identity MUST have complete immutable mandatory source coverage.
missing=copy.deepcopy(c._pins);del missing['sources']['review_pc_moments/reference.py']
try:P.verify_pins(missing);accepted=True
except ValueError:accepted=False
check('missing loaded PC source pin rejected',not accepted)
# Real owned mirror: remove PC pin, alter PC source bytes, recompute caller identity.
mirror=OWN/'missing_pc_mirror'
for name in c._pins['sources']:
 target=mirror/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
f=mirror/'review_pc_moments/reference.py';f.write_text(f.read_text()+'\n# independent review: source changed but omitted from pins\n')
(mirror/'isolated_microphysics/PINS.json').write_text(P.dumps(missing))
req=copy.deepcopy(rows[0]['request']);req['identity']=P.digest(missing)
done=subprocess.run([sys.executable,'-I','-B',str(mirror/'isolated_microphysics/worker_entry.py')],input=P.dumps(req),text=True,capture_output=True,timeout=110,cwd=OWN)
try:mirrorout=P.loads(done.stdout)
except Exception:mirrorout=dict(raw=done.stdout)
check('actual unpinned changed PC source rejected before evaluation',done.returncode!=0,dict(returncode=done.returncode,status=mirrorout.get('status'),identity=mirrorout.get('identity')))
# Wire schema says actual channel totals are positive, never empty inverse zeros.
mut=copy.deepcopy(rows[0]['response']);req=rows[0]['request']
for key in ['number','energy']:
 mut['sources']['channels']['positron_capture'][key]=copy.deepcopy(mut['sources']['total'][key]);mut['sources']['channels']['neutron_beta'][key]=P.represented(None,zero_reason='empty_incident_radiation')
try:c.validate(mut,req);accepted=True
except (ValueError,RuntimeError,ZeroDivisionError):accepted=False
check('false exact-zero beta emission rejected',not accepted)
# False booleans must not pass as numeric exact zero.
mut=copy.deepcopy(rows[0]['response']);mut['sources']['inverse']['value']=False
try:c.validate(mut,req);accepted=True
except (ValueError,RuntimeError,ZeroDivisionError):accepted=False
check('boolean inverse zero rejected',not accepted)
for log,reason in [(-55933.,'underflow'),(1000.,'overflow')]:
 x=P.represented(log);P.validate_positive(x);check('finite log retained '+reason,x['log']==log and x['value']is None and x['reason']==reason)
check('all frozen source and raw bytes unchanged',before==snapshot())
out=dict(checks=checks,passed=sum(x['passed']for x in checks),failed=[x for x in checks if not x['passed']],calls=rows,changed_unpinned_source_response=mirrorout,frozen_before=before,frozen_after=snapshot(),seconds=time.monotonic()-start)
(OWN/'initial_review_results.json').write_text(json.dumps(out,indent=2));print(json.dumps(dict(passed=out['passed'],failed=out['failed'],seconds=out['seconds']),indent=2))
