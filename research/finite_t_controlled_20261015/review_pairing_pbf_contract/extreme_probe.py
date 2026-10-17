from pathlib import Path
import importlib.util,sys,json,math,time,warnings
import numpy as np
from scipy.special import zeta
P=Path(__file__).resolve().parent;A=P.parent/'pairing_pbf_contract';sys.dont_write_bytecode=True
s=importlib.util.spec_from_file_location('pbf_extreme_q',A/'quasiparticle.py');q=importlib.util.module_from_spec(s);sys.modules[s.name]=q;s.loader.exec_module(q)
s=importlib.util.spec_from_file_location('pbf_initial_q',A/'initial_small_gap_quasiparticle.py');old=importlib.util.module_from_spec(s);sys.modules[s.name]=old;s.loader.exec_module(old)
start=time.monotonic();checks=[];rows=[]
try:old.log_F_B(1e-200);bad=False
except ValueError as e:bad=True;failure=str(e)
checks.append(dict(name='reproduce preserved old tinygap failure',passed=bad,error_message=failure if bad else None))
coef=24*((1-2**-4)*zeta(5)-(1-2**-3)*zeta(4))
for v in [1e-280,1e-307,1e-320,float(np.nextafter(0.,1.))]:
 with warnings.catch_warnings(record=True)as ws:
  a=q.log_F_A(v);b=q.log_F_B(v)
 expected=2*math.log(v)+math.log(2*coef);err=math.expm1(b-expected)
 rows.append(dict(v=v,logFA=a,logFB=b,analytic_logFB=expected,relative_error=err,warnings=[str(x.message)for x in ws],status='selectedpass'if abs(err)<3e-10 else'excluded subnormal-input boundary extension; not an author selected benchmark'))
 checks.append(dict(name='FB tinygap analytic '+repr(v),passed=abs(err)<3e-10,error=err,tolerance=3e-10,classification='selected'if v>=1e-307 else'new boundary-extension probe'))
out=dict(checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],rows=rows,seconds=time.monotonic()-start)
(P/'extreme_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
