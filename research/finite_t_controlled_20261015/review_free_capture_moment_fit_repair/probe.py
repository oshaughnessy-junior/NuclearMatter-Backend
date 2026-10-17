"""Independent narrow repair recheck, no physical-grid replay or author edits."""
from pathlib import Path
import sys,math,json,hashlib,importlib.util,copy
import numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parent;A=ROOT/'free_capture_moment_fit';sys.dont_write_bytecode=True
sp=importlib.util.spec_from_file_location('fit_qualification_independent_repaired',A/'approximation.py');F=importlib.util.module_from_spec(sp);sys.modules[sp.name]=F;sp.loader.exec_module(F)
old=type(sys)('fit_qualification_independent_preserved');old.__file__=str(A/'approximation.py');sys.modules[old.__name__]=old;exec(compile((A/'pre_qualification_repair/approximation.py').read_text(),str(A/'pre_qualification_repair/approximation.py'),'exec'),old.__dict__)
checks=[];rows=[];before={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in A.rglob('*')if p.is_file()};initial_review=ROOT/'review_free_capture_moment_fit'
def ck(n,v):checks.append(dict(name=n,passed=bool(v)))
ck('repaired API source hash',hashlib.sha256((A/'approximation.py').read_bytes()).hexdigest()=='9020fb5dab238e2b7c169612b037750c04dfe19932f22976470f806b2514d3b3');ck('repaired manifest hash',before['free_capture_moment_fit/MANIFEST.json']=='bdcff1c4837be93d5d94c63c5b165d6d7e0a1dbd17e9c94a8edfa513348008cc')
rb=Path.read_bytes;rt=Path.read_text;states=json.loads((A/'REVISED_LOCK2.json').read_text());allstates=states['training']+states['heldout']+json.loads((initial_review/'VALIDATION_LOCK.json').read_text())['states']
for i,s in enumerate(allstates):
 a=old.approximate_moments(**s);b=F.approximate_moments(**s);ck('old/new complete return bitwise '+str(i),json.dumps(a,separators=(',',':')).encode()==json.dumps(b,separators=(',',':')).encode());rows.append(dict(state=s,old=a,repaired=b))
# Every byte interface supplies the same forged report, reproducing old witnesses.
realq=json.loads((A/'HELDOUT_ASSESSMENT.json').read_text());cases=[]
for witness in json.loads((initial_review/'guard_results.json').read_text())['reproducers']:cases.append((witness['name'],'HELDOUT_ASSESSMENT.json',json.dumps(witness['qualification']).encode(),True))
for name,update in [('missing checks',lambda q:q.pop('checks')),('failed gate',lambda q:q['checks'][0].update(passed=False,error=.5)),('stale coeff',lambda q:q.update(coefficient_sha256='0'*64)),('missing last row',lambda q:q['rows'].pop()),('wrong row state',lambda q:q['rows'][0]['state'].update(T=2.99)),('nonfinite error',lambda q:q['checks'][0].update(error=float('nan'))),('retuned',lambda q:q.update(basis_retuned=True)),('nonliteral true',lambda q:q.update(qualified=1))]:
 q=copy.deepcopy(realq);update(q);cases.append((name,'HELDOUT_ASSESSMENT.json',json.dumps(q).encode(),False))
for rel in ['HELDOUT_ASSESSMENT.json','COEFFICIENT_FREEZE.json','REVISED_LOCK2.json','SOURCE_PIN.json','heldout_0.json','heldout_7.json','training_0.json','training_10.json']:
 cases.append(('missing '+rel,rel,None,False));cases.append(('altered '+rel,rel,rb(A/rel)+b' ',False))
cases.append(('wrong physical current','../muon_full_current/current.py',b'changed source',False))
match=F.M.EC.matched;prep=[]
def spy(*a,**kw):prep.append(True);raise AssertionError('EOS preparation reached on malformed evidence')
F.M.EC.matched=spy
for name,rel,payload,old_witness in cases:
 target=(A/rel).resolve()
 def bytes_read(p):
  if p.resolve()==target:
   if payload is None:raise FileNotFoundError('Synthetic missing evidence')
   return payload
  return rb(p)
 def text_read(p,*a,**kw):
  if p.resolve()==target:
   if payload is None:raise FileNotFoundError('Synthetic missing evidence')
   return payload.decode()
  return rt(p,*a,**kw)
 Path.read_bytes=bytes_read;Path.read_text=text_read;prep.clear()
 try:F.approximate_moments(1.6,4e-7,.32);rejected=False
 except RuntimeError:rejected=True
 finally:Path.read_bytes=rb;Path.read_text=rt
 ck('reject before EOS '+name,rejected and not prep)
 if old_witness:
  Path.read_bytes=bytes_read;Path.read_text=text_read
  try:old.approximate_moments(1.6,4e-7,.32);accepted=True
  except Exception:accepted=False
  finally:Path.read_bytes=rb;Path.read_text=rt
  ck('preserved old witness still accepts '+name,accepted)
F.M.EC.matched=match
# Hash/parse must use one read of each actual document. Subsequent read would
# serve poison, and any text read of a critical report is prohibited.
critical={A/n for n in ['HELDOUT_ASSESSMENT.json','COEFFICIENT_FREEZE.json','SOURCE_PIN.json','REVISED_LOCK2.json']};counts={str(p.name):0 for p in critical}
def single_bytes(p):
 if p in critical:
  counts[p.name]+=1
  if counts[p.name]>1:return b'{"qualified":false,"race":true}'
 return rb(p)
def no_critical_text(p,*a,**kw):
 if p in critical:raise AssertionError('Separate parse read would consume unchecked bytes')
 return rt(p,*a,**kw)
Path.read_bytes=single_bytes;Path.read_text=no_critical_text
try:v=F.approximate_moments(1.6,4e-7,.32);works=True
finally:Path.read_bytes=rb;Path.read_text=rt
ck('same consumed bytes no double-read race',works and all(v==1 for v in counts.values()))
q=json.loads(rb(A/'HELDOUT_ASSESSMENT.json'));f=json.loads(rb(A/'COEFFICIENT_FREEZE.json'));ck('actual report32complete1percent gates',len(q['checks'])==32 and all(c['passed']is True and c['gate']==.01 and abs(c['error'])<=.01 for c in q['checks']));ck('actual report8ordered rows',len(q['rows'])==8 and all(r['index']==i and r['state']==states['heldout'][i]for i,r in enumerate(q['rows'])));ck('actual report same coeff pin',q['coefficient_sha256']==F.COEFFICIENT_SHA)
for i,r in enumerate(q['rows']):ck('actual heldout raw hash '+str(i),hashlib.sha256(rb(A/f'heldout_{i}.json')).hexdigest()==r['raw_sha256'])
for n,h in f['training_files'].items():ck('training current hash '+n,hashlib.sha256(rb(A/n)).hexdigest()==h)
# Frozen records prior to repair remain exact, including initial review failures.
pin=json.loads((A/'PRE_QUALIFICATION_REPAIR_PIN.json').read_text())
# Accommodate explicit manifest field names only after inspecting them below.
for n,h in before.items():
 if n.startswith('free_capture_moment_fit/pre_qualification_repair/')or n.startswith('free_capture_moment_fit/pre_qualification_repair_review/'):
  ck('preserved snapshot '+n,hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h)
for name,h in json.loads((initial_review/'CHECKPOINT.json').read_text())['files'].items():ck('initial review immutable '+name,hashlib.sha256((initial_review/name).read_bytes()).hexdigest()==h)
after={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in A.rglob('*')if p.is_file()};ck('all author bytes unchanged',before==after)
out=dict(checks=checks,passed=sum(x['passed']for x in checks),failed=[x for x in checks if not x['passed']],rows=rows,single_read_counts=counts,replayed_cases=[n for n,_,_,_ in cases],before=before,after=after);(P/'results.json').write_text(json.dumps(out,indent=2)+'\n');print(out['passed'],out['failed'])
