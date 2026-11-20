from pathlib import Path
import importlib.util,sys,json,copy,math,hashlib
import numpy as np
P=Path(__file__).resolve().parent;A=P.parent/'free_capture_moment_fit';sys.dont_write_bytecode=True
s=importlib.util.spec_from_file_location('review_optional_fit_guard_only',A/'approximation.py');F=importlib.util.module_from_spec(s);sys.modules[s.name]=F;s.loader.exec_module(F);checks=[];failures=[]
def ck(n,ok):checks.append(dict(name=n,passed=bool(ok)))
for state in [(.99,4e-7,.32),(3.01,4e-7,.32),(2.,.99e-7,.32),(2.,1.001e-6,.32),(2.,4e-7,.199),(2.,4e-7,.501),(1.,1e-6,.5),(True,4e-7,.32),(1.6,False,.32),(1.6,4e-7,True),(float('nan'),4e-7,.32),(1.6,float('inf'),.32),(1.6,4e-7,np.array(.32))]:
 try:F.approximate_moments(*state);ok=False
 except ValueError:ok=True
 ck('reject invalid/outside intersection '+repr(state),ok)
qual=A/'HELDOUT_ASSESSMENT.json';original=Path.read_text;real=json.loads(qual.read_text());matched=F.M.EC.matched;preparation=[]
def spy(*a,**kw):preparation.append(True);raise AssertionError('Physical preparation reached')
F.M.EC.matched=spy
for name,value in [('qualified false',dict(real,qualified=False)),('qualified integer',dict(real,qualified=1)),('wrong coefficient',dict(real,coefficient_sha256='0'*64))]:
 Path.read_text=lambda p,*a,**kw:json.dumps(value)if p==qual else original(p,*a,**kw)
 try:F.approximate_moments(1.6,4e-7,.32);good=False
 except RuntimeError:good=True
 finally:Path.read_text=original
 ck('reject prior to EOS '+name,good and not preparation)
F.M.EC.matched=matched
# No frozen file is changed. Return current-style metadata with actual failed
# heldout checks while retaining the true coefficient association/qualified flag.
for name,fake in [('failed heldout with stale true flag',dict(real,checks=[dict(name='failure',error=.5,gate=.01,passed=False)],passed=0,failed=[dict(name='failure',passed=False)])),('empty fabricated qualification',dict(qualified=True,coefficient_sha256=F.COEFFICIENT_SHA))]:
 Path.read_text=lambda p,*a,**kw:json.dumps(fake)if p==qual else original(p,*a,**kw)
 try:r=F.approximate_moments(1.6,4e-7,.32);good=False;reason='ACCEPTED positive approximation despite invalid qualification'
 except (RuntimeError,ValueError,KeyError):good=True;reason='rejected'
 finally:Path.read_text=original
 ck('reject malformed qualification '+name,good);failures.append(dict(name=name,qualification=fake,actual=reason))
out=dict(checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],reproducers=failures,scope='In-memory changed external qualification metadata; no source/file mutation or untrusted-code security claim.');(P/'guard_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out['passed'],out['failed'])
