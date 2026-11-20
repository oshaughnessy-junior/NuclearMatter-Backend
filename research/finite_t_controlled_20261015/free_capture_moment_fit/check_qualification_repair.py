"""Author narrow admission repair tests; no physical data regeneration."""
from pathlib import Path
import sys,types,json,hashlib,copy,contextlib
from unittest.mock import patch
P=Path(__file__).resolve().parent;sys.dont_write_bytecode=True
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def load(name,text):
 m=types.ModuleType(name);m.__file__=str(P/'approximation.py');sys.modules[name]=m;exec(compile(text,str(P/'approximation.py'),'exec'),m.__dict__);return m
old=load('before_qualification_repair_only',(P/'pre_qualification_repair/approximation.py').read_text());new=load('after_qualification_repair_only',(P/'approximation.py').read_text());checks=[]
def ck(n,v):checks.append(dict(name=n,passed=bool(v)))
qual=P/'HELDOUT_ASSESSMENT.json';real=json.loads(qual.read_text());readbytes=Path.read_bytes;readtext=Path.read_text
@contextlib.contextmanager
def substitute(changes):
 def b(path,*a,**kw):
  if path in changes:
   if changes[path]is None:raise FileNotFoundError(str(path))
   return changes[path]
  return readbytes(path,*a,**kw)
 def t(path,*a,**kw):
  if path in changes:return b(path).decode('utf-8')
  return readtext(path,*a,**kw)
 with patch.object(Path,'read_bytes',b),patch.object(Path,'read_text',t):yield
review=json.loads((P/'pre_qualification_repair_review/guard_results.json').read_text())
for i,reproducer in enumerate(review['reproducers']):
 payload=json.dumps(reproducer['qualification']).encode()
 with substitute({qual:payload}):
  try:old.approximate_moments(1.6,4e-7,.32);accepted=True
  except Exception:accepted=False
 ck('exact initial witness accepted by old'+str(i),accepted)
variants=[('exact initial witness '+str(i),{qual:json.dumps(r['qualification']).encode()})for i,r in enumerate(review['reproducers'])]
for name,change in [('false',dict(qualified=False)),('boolalias',dict(qualified=1)),('wrong coefficient',dict(coefficient_sha256='0'*64)),('missing row',dict(rows=real['rows'][:-1])),('missing check',dict(checks=real['checks'][:-1])),('altered gate',dict(checks=[dict(c,gate=.02)for c in real['checks']])),('stale source',dict(rows=[dict(r,raw_sha256='0'*64)for r in real['rows']])),('retuned',dict(basis_retuned=True)),('incomplete passed',dict(passed=31))]:
 variants.append((name,{qual:json.dumps(dict(real,**change)).encode()}))
variants.extend([('altered sourcepins',{P/'SOURCE_PIN.json':b'{}'}),('missing sourcepins',{P/'SOURCE_PIN.json':None}),('altered lockedstatecontract',{P/'REVISED_LOCK2.json':b'{}'}),('missing report',{qual:None}),('unparsable report',{qual:b'{bad'}),('whitespace-altered report',{qual:readbytes(qual)+b' '}),('missing source0',{P/'heldout_0.json':None}),('altered source0',{P/'heldout_0.json':b'{"qualified":true}'}),('missing training0',{P/'training_0.json':None}),('altered coefficients',{P/'COEFFICIENT_FREEZE.json':b'{"coefficients":[]}'}),('source code changed',{P.parent/'weak_rates/static_capture.py':b'not the physical source'})])
for name,changes in variants:
 prepared=[]
 def spy(*a,**kw):prepared.append(True);raise AssertionError('EOS should not be reached')
 with substitute(changes),patch.object(new.M.EC,'matched',spy):
  try:new.approximate_moments(1.6,4e-7,.32);rejected=False
  except RuntimeError:rejected=True
 ck('reject beforeEOS '+name,rejected and not prepared)
# All19 exact state API outputs must remain bitwise equal to initial implementation.
for group,n in [('training',11),('heldout',8)]:
 for i in range(n):
  state=json.loads((P/f'{group}_{i}.json').read_text())['state'];a=old.approximate_moments(**state);b=new.approximate_moments(**state)
  ck('complete API bitwise '+str((group,i)),json.dumps(a)==json.dumps(b))
# Four independent review points, no detailed phase recomputation.
for st in [(1.6,4e-7,.32),(2.8,8e-7,.23),(1.87,7e-7,.41),(2.37,3e-7,.27)]:ck('independentpoint old/new '+str(st),old.approximate_moments(*st)==new.approximate_moments(*st))
pin=json.loads((P/'PRE_QUALIFICATION_REPAIR_PIN.json').read_text())
for name,h in pin['initial_author'].items():
 ck('initial author snapshot '+name,sha(P/'pre_qualification_repair'/name)==h)
 if name not in['approximation.py','MANIFEST.json']:ck('physical/model/artifact unchanged '+name,sha(P/name)==h)
for name,h in pin['initial_review'].items():
 ck('initial review snapshot '+name,sha(P/'pre_qualification_repair_review'/name)==h)
 ck('initial reviewer files untouched '+name,sha(P.parent/'review_free_capture_moment_fit'/name)==h)
# Remove only guards from both versions: the valid physical arithmetic suffix is exact.
a=(P/'pre_qualification_repair/approximation.py').read_text();b=(P/'approximation.py').read_text();a=a[a.index(' b,st=M.EC.matched'):];b=b[b.index(' b,st=M.EC.matched'):]
a=a.replace(" f=json.loads((P/'COEFFICIENT_FREEZE.json').read_text());base=[]"," base=[]").replace("coefficient_sha256=hashlib.sha256((P/'COEFFICIENT_FREEZE.json').read_bytes()).hexdigest()","coefficient_sha256=COEFFICIENT_SHA")
ck('physical arithmetic suffix unchanged afteratomic-read substitutions',a==b)
out=dict(status='AUTHOR_GUARD_REPAIR_TESTS_NOT_INDEPENDENT_ACCEPTANCE',checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],physical_grid_rerun=False,source_sha256=sha(P/'approximation.py'),actual_report_sha256=sha(qual))
(P/'qualification_repair_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out['passed'],out['failed'])
