"""Independent repaired-contract recheck; initial failed evidence untouched."""
from pathlib import Path
import sys,types,importlib.util,json,hashlib,copy,subprocess,shutil,time,math
ROOT=Path(__file__).resolve().parents[1];OWN=Path(__file__).parent;LEAF=ROOT/'isolated_microphysics';sys.dont_write_bytecode=True
checks=[];start=time.monotonic()
def ck(n,ok,details=None):checks.append(dict(name=n,passed=bool(ok),details=details))
def rejects(n,fn):
 try:fn();ok=False
 except (ValueError,RuntimeError,TypeError,ZeroDivisionError,OverflowError):ok=True
 ck(n,ok)
def hashes(paths):return{str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths if p.is_file()and'__pycache__'not in str(p)}
initial_paths=[OWN/'probe.py',OWN/'REVIEW.md',OWN/'CHECKPOINT.json',OWN/'initial_review_results.json']+list((OWN/'missing_pc_mirror').rglob('*'));initial=hashes(initial_paths)
dirs=['isolated_microphysics','transparent_antinu_element','hot_beta_moments','review_pc_moments','relativistic_crossed_contract','relativistic_crossed_thermal','relativistic_free_eos','thermodynamics','relativistic_tensor','relativistic_capture_oracle','muon_full_current','muon_vacuum'];paths=[p for d in dirs for p in(ROOT/d).rglob('*')];before=hashes(paths)
sentinels={}
for n in ['study','protocol','hot_moments','kernel','beta','vacuum_cc','current','events','free_baryons','lepton_reference']:
 sentinels[n]=types.ModuleType(n);sys.modules[n]=sentinels[n]
sp=importlib.util.spec_from_file_location('review_repaired_isolated_client',LEAF/'client.py');C=importlib.util.module_from_spec(sp);sys.modules[sp.name]=C;sp.loader.exec_module(C);c=C.Client();P=C.P
ck('exactly13 declared required sources',len(P.REQUIRED_SOURCES)==13 and set(c._pins['sources'])==P.REQUIRED_SOURCES)
ck('commissioned repaired manifest SHA',hashlib.sha256((LEAF/'MANIFEST.json').read_bytes()).hexdigest()=='704534d7dd3378f61fbec236fc375e0dce679b533d3567426f38d9da2c21651b')
basepath=list(sys.path);basemod=dict(sys.modules);calls=[]
for T,Y,profile in [(2.,.3,'base'),(1.9955690477112278,.3218092326059971,'fine'),(2.,.3,'fine')]:
 s=C.MaterialState(T,1e-7,Y);ctl=C.Controls(profile);req=c.request(s,ctl);out=c.evaluate(s,ctl);calls.append(dict(request=req,response=out))
 ck('parent collision identities '+str((T,profile)),all(sys.modules[n]is m for n,m in sentinels.items()))
 ck('root charge independently reconstructed '+str((T,profile)),abs((out['eos']['densities']['electron']-out['eos']['densities']['positron'])/(1e-7*Y)-1)<2e-9)
 if T==2.:R,Q=1.1173618686632175e-6,1.1994635801727575e-5;tol=1e-8
 else:R,Q=1.0642163330393762e-6,1.1400114218759336e-5;tol=2e-12
 for key,ref in [('number',R),('energy',Q)]:ck('fresh physical forward '+str((T,profile,key)),abs(out['sources']['total'][key]['value']/ref-1)<tol)
ck('parent sys.path unchanged',sys.path==basepath);ck('no replaced parent modules',all(sys.modules.get(n)is m for n,m in basemod.items()));ck('no new parent modules',set(sys.modules)==set(basemod))
# A fixed independently enumerated source set, not just the author constant count.
expected={'hot_beta_moments/hot_moments.py','relativistic_crossed_contract/beta.py','relativistic_crossed_thermal/kernel.py','relativistic_free_eos/free_baryons.py','thermodynamics/lepton_reference.py','relativistic_tensor/tensor.py','relativistic_capture_oracle/events.py','muon_full_current/current.py','muon_vacuum/vacuum_cc.py','review_pc_moments/reference.py','isolated_microphysics/client.py','isolated_microphysics/protocol.py','isolated_microphysics/worker_entry.py'}
ck('reviewer independently enumerated transitive source set',set(c._pins['sources'])==expected)
for name in expected:
 z=copy.deepcopy(c._pins);del z['sources'][name];rejects('missing required source '+name,lambda z=z:P.verify_pins(z))
z=copy.deepcopy(c._pins);z['sources']['isolated_microphysics/check_interface.py']='0'*64;rejects('extra driver source rejected',lambda:P.verify_pins(z))
for sha in ['0'*63,'x'*64,False,None,float('nan')]:
 z=copy.deepcopy(c._pins);z['sources']['review_pc_moments/reference.py']=sha;rejects('invalid digest '+repr(sha),lambda z=z:P.verify_pins(z))
# Actual changed source and a top-level execution marker, with/without mandatory pin.
mirrors=[]
for missing in [False,True]:
 mirror=OWN/('repaired_missing_pc_mirror'if missing else'repaired_changed_pc_mirror');marker=mirror/'UNEXPECTED_EXECUTION.txt'
 if marker.exists():raise RuntimeError('Refusing to erase existing marker evidence')
 for name in expected:
  dst=mirror/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dst)
 f=mirror/'review_pc_moments/reference.py';f.write_text(f.read_text()+f'\nfrom pathlib import Path as _review_path\n_review_path({str(marker.resolve())!r}).write_text("UNPINNED_SOURCE_EXECUTED")\n')
 pins=copy.deepcopy(c._pins)
 if missing:del pins['sources']['review_pc_moments/reference.py']
 (mirror/'isolated_microphysics/PINS.json').write_text(P.dumps(pins));req=copy.deepcopy(calls[0]['request']);req['identity']=P.digest(pins)
 done=subprocess.run([sys.executable,'-I','-B',str(mirror/'isolated_microphysics/worker_entry.py')],input=P.dumps(req),text=True,capture_output=True,timeout=10,cwd=OWN);out=P.loads(done.stdout)
 ck('changed PC preexecution rejects missing='+str(missing),done.returncode==2 and out.get('status')=='error'and not done.stderr and not marker.exists(),out)
 mirrors.append(dict(missing=missing,marker_absent=not marker.exists(),returncode=done.returncode,response=out))
rejects('spec-loaded wrong origin rejected',lambda:P.verify_file(LEAF/'client.py','review_pc_moments/reference.py',c._pins))
# Branch-specific positive admission: coherently reassign combined totals to the other channel.
out,req=calls[0]['response'],calls[0]['request']
for branch,other in [('neutron_beta','positron_capture'),('positron_capture','neutron_beta')]:
 for fields in [('number',),('energy',),('number','energy')]:
  z=copy.deepcopy(out)
  for field in fields:
   z['sources']['channels'][branch][field]=P.represented(None,zero_reason='empty_incident_radiation');z['sources']['channels'][other][field]=copy.deepcopy(z['sources']['total'][field])
  rejects('false emission zero '+branch+str(fields),lambda z=z:c.validate(z,req))
for val in [False,True,'0',None,float('nan'),float('inf'),-float('inf')]:
 z=copy.deepcopy(out);z['sources']['inverse']['value']=val;rejects('inverse invalid zero '+repr(val),lambda z=z:c.validate(z,req))
for field in ['value','log']:
 for val in [False,True,float('nan'),float('inf')]:
  z=copy.deepcopy(out);z['sources']['channels']['neutron_beta']['number'][field]=val;rejects('emission invalid '+field+repr(val),lambda z=z:c.validate(z,req))
for val in [False,float('nan'),float('inf')]:
 z=copy.deepcopy(out);z['eos']['density_residual']=val;rejects('root invalid '+repr(val),lambda z=z:c.validate(z,req))
for val in [0,0.]:
 z=copy.deepcopy(out);z['sources']['inverse']['value']=val
 try:c.validate(z,req);ok=True
 except Exception:ok=False
 ck('actual numeric inverse zero admitted '+repr(val),ok)
for log,reason in [(-55933.,'underflow'),(1000.,'overflow')]:
 r=P.represented(log);P.validate_positive(r);ck('unrepresented finite log '+reason,r['value']is None and r['log']==log and r['reason']==reason)
# Other independent request/output attacks remain rejected, not reinterpreted as new source physics.
for key,val in [('T',True),('T',float('nan')),('Ye',False),('nB',2e-7)]:
 s=dict(T=2.,nB=1e-7,Ye=.3);s[key]=val;rejects('bad request state '+key+repr(val),lambda s=s:c.request(C.MaterialState(**s)))
for name,mut in [('weakmass',lambda z:z['request']['conventions']['weak_parameters'].update(bare_weak_mass_MeV=939.)),('units',lambda z:z['request']['conventions'].update(units='cgs')),('controls',lambda z:z['request']['controls']['pc'].__setitem__(0,29)),('signedLe',lambda z:z['sources']['signed'].update(material_Le=-1)),('totalenergy',lambda z:z['sources']['total']['energy'].update(value=float('inf'))),('identity',lambda z:z.update(identity='0'*64))]:
 z=copy.deepcopy(out);mut(z);rejects('malformed response '+name,lambda z=z:c.validate(z,req))
ck('initial14/4 raw/source/mirror bytes preserved',initial==hashes(initial_paths));ck('all frozen author/upstream source/raw unchanged',before==hashes(paths))
r=dict(checks=checks,passed=sum(x['passed']for x in checks),failed=[x for x in checks if not x['passed']],calls=calls,mirrors=mirrors,source_pins=c._pins,initial_evidence=initial,frozen_before=before,frozen_after=hashes(paths),seconds=time.monotonic()-start)
(OWN/'repair_results.json').write_text(json.dumps(r,indent=2));print(json.dumps(dict(passed=r['passed'],failed=r['failed'],seconds=r['seconds']),indent=2))
