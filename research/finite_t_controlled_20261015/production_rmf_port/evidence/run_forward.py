from pathlib import Path
import sys,importlib.util,json,time,hashlib
P=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True
sp=importlib.util.spec_from_file_location('rmf_independent_forward_tests',P/'staged/tests/test_finite_t_rmf_forward.py');O=importlib.util.module_from_spec(sp);sp.loader.exec_module(O)
# Independent originals are all computed before importing or evaluating the candidate.
t0=time.monotonic();points=O.POINTS;oracles=[O.oracle(*p)for p in points]
(P/'evidence/independent_forward.json').write_text(json.dumps(dict(rows=oracles,seconds=time.monotonic()-t0,candidate_imported=False),indent=2)+'\n')
sys.path.insert(0,str(P/'test_mirror/src'))
from nmbackend.reference.finite_t_rmf import state_density
checks=[];rows=[]
for inp,ex in zip(points,oracles):
 start=time.monotonic();s=state_density(*inp);row=s.asdict();rows.append(row)
 for key in ['phi','dirac_mass','mu_n','mu_p','kinetic_mu_n','kinetic_mu_p','energy_density','pressure','entropy_density','free_energy_density','neutron_density','proton_density','scalar_density']:
  actual=getattr(s,key);err=abs(actual-ex[key])/max(abs(ex[key]),1e-20);checks.append(dict(point=inp,field=key,error=err,gate=2e-9,passed=bool(err<=2e-9)))
 (P/'evidence/forward_partial.json').write_text(json.dumps(dict(checks=checks,states=rows),indent=2)+'\n');print(inp,'candidate seconds',time.monotonic()-start,flush=True)
out=dict(passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],states=rows,checks=checks,seconds=time.monotonic()-t0)
(P/'evidence/forward_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out['passed'],out['failed'],out['seconds'])
