"""Bounded consumer admission and preexecution adversaries, frozen bytes untouched."""
from pathlib import Path
import json,importlib.util,sys,copy,hashlib
import numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parent;sys.dont_write_bytecode=True
s=importlib.util.spec_from_file_location('balance_review_control_unique',ROOT/'transparent_capture_balance/balance.py');M=importlib.util.module_from_spec(s);sys.modules[s.name]=M;s.loader.exec_module(M)
checks=[]
def reject(name,fn,types=(ValueError,RuntimeError)):
 try:fn();error='NO REJECTION';ok=False
 except types as e:error=str(e);ok=True
 except Exception as e:error=type(e).__name__+':'+str(e);ok=False
 checks.append(dict(name=name,passed=ok,result=error))
for x in [float('nan'),float('inf'),-float('inf'),True,False,np.bool_(True),None,'0.5',0,.049,.901,1,complex(.5)]:reject('material invalid '+repr(x),lambda x=x:M.material(x))
for x in [0,1,None,np.bool_(True),'fine',float('nan')]:reject('refined invalid '+repr(x),lambda x=x:M.evaluate(.3,x))
original=M.HERE;material=M.material
scratch=P/'admission_scratch';scratch.mkdir(exist_ok=True)
(scratch/'UPSTREAM.json').write_text((original/'UPSTREAM.json').read_text())
a=json.loads((original/'EC_REVIEW_ACCEPTANCE.json').read_text())
M.HERE=scratch
# Spy makes this a preexecution check, not mere exception anywhere in scientific call.
def forbidden(y):raise AssertionError('SOURCE PREPARATION EXECUTED')
M.material=forbidden
reject('absent independent acceptance blocks before EOS/source',lambda:M.evaluate(.3))
for name,change in [('accepted false',{'accepted':False}),('accepted integer',{'accepted':1}),('wrong EC source hash',{'source_sha256':'0'*64}),('changed review hash',{'review_sha256':'0'*64})]:
 b=dict(a);b.update(change);(scratch/'EC_REVIEW_ACCEPTANCE.json').write_text(json.dumps(b));reject(name+' preexecution',lambda:M.evaluate(.3))
(scratch/'EC_REVIEW_ACCEPTANCE.json').write_text(json.dumps(a));old=M.PINS['hot_electron_capture_moments/electron_capture.py'];M.PINS['hot_electron_capture_moments/electron_capture.py']='0'*64
reject('altered pinned EC rejected before EOS/source',lambda:M.evaluate(.3));M.PINS['hot_electron_capture_moments/electron_capture.py']=old
# Pin overlay and accepted explicit copy of both same-source review controls.
M.HERE=original;M.material=material;M.require_acceptance()
checks.append(dict(name='current source matched acceptance succeeds',passed=True,result='source SHA/review SHA/upstream checks pass'))
for y in [.05,.9]:
 b,state=M.material(y);checks.append(dict(name='admitted endpoint finite '+str(y),passed=all(np.isfinite([b.mu_n,b.mu_p,b.mu_e])),result='EOS only; no endpoint source/domain certificate'))
(P/'controls.json').write_text(json.dumps(dict(checks=checks,passed=sum(bool(c['passed'])for c in checks),failed=[c for c in checks if not c['passed']]),indent=2,default=lambda x:bool(x)if isinstance(x,np.bool_)else str(x))+'\n')
print(len(checks),[c for c in checks if not c['passed']])
