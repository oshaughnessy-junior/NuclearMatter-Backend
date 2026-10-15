"""Actual tiny-kernel cache replay and deliberate provenance mismatches."""
from pathlib import Path
import tempfile,json,copy,time,subprocess,sys
import cache_contract as c
import integrated as i
P=Path(__file__).resolve().parent;start=time.monotonic();checks=[]
def gate(name,value):checks.append(dict(name=name,passed=bool(value)))
def reject(name,path,expected):
 try:c.load(path,expected)
 except ValueError:gate(name,True)
 else:gate(name,False)
m=i.slab.matter_state(2.,7e-5,3e-5);opts=dict(outer_order=4,internal_order=4,edges=(0.,.1))
request=c.fingerprint(m,'nue',**opts);key=c.digest(request);expected={key:request}
actual=i.integrate(m,'nue',**opts);row=c.record(request,actual)
with tempfile.TemporaryDirectory() as td:
 path=Path(td)/'cache.json';c.save(path,[row],.1);replay=c.load(path,expected)
 gate('actual tiny physics result exact replay',replay[key]==actual)
 for name,kwargs in [('outer order',dict(outer_order=5)),('internal order',dict(internal_order=5)),('internal tail',dict(tail=41.)),('edges',dict(edges=(0.,.2))),('length',dict(length_cm=2e5))]:
  kw=opts|kwargs;r=c.fingerprint(m,'nue',**kw);reject('reject changed '+name,path,{c.digest(r):r})
 r=c.fingerprint(m,'antinue',**opts);reject('reject flavor',path,{c.digest(r):r})
 for name,args in [('T',(2.01,7e-5,3e-5)),('nn',(2.,7.01e-5,3e-5)),('np',(2.,7e-5,3.01e-5))]:
  r=c.fingerprint(i.slab.matter_state(*args),'nue',**opts);reject('reject state '+name,path,{c.digest(r):r})
 for name,kw in [('mass',dict(mass=940.)),('offset',dict(offset_neutron=1.4))]:
  r=c.fingerprint(i.slab.matter_state(2.,7e-5,3e-5,**kw),'nue',**opts);reject('reject state '+name,path,{c.digest(r):r})
 for source in request['source_sha256']:
  r=copy.deepcopy(request);r['source_sha256'][source]='0'*64;reject('reject source '+source,path,{c.digest(r):r})
 r=copy.deepcopy(request);r['environment']['numpy']='other';reject('reject environment',path,{c.digest(r):r})
 data=json.loads(path.read_text());data['records'][0]['result']['values']['emission_number_cm3_s']*=2;path.write_text(json.dumps(data));reject('reject result corruption',path,expected)
 c.save(path,[row,row],.1);reject('reject duplicate entries',path,expected)
 path.write_text(json.dumps({'runs':[actual]}));reject('reject missing fingerprint legacy cache',path,expected)
 reject('reject missing file',Path(td)/'absent.json',expected)
# Run the actual production entry against a legacy partial. It must fail before
# any physical integration and leave the legacy file byte-for-byte unchanged.
legacy=P/'dilute_partial.json';before=legacy.read_bytes();proc=subprocess.run([sys.executable,str(P/'run_study.py'),'dilute','--resume'],capture_output=True,text=True)
gate('actual entry rejects legacy before kernel',proc.returncode!=0 and 'Missing exact cache provenance' in proc.stderr)
gate('rejected legacy file unchanged',legacy.read_bytes()==before)
report=dict(passed=all(x['passed'] for x in checks),checks=checks,runtime_seconds=time.monotonic()-start,entry_stdout=proc.stdout,entry_stderr=proc.stderr,source_sha256=request['source_sha256'],actual_tiny_result=actual)
(P/'cache_contract_results.json').write_text(json.dumps(report,indent=2)+'\n');print(sum(x['passed'] for x in checks),'/',len(checks),report['runtime_seconds']);raise SystemExit(0 if report['passed'] else 1)
