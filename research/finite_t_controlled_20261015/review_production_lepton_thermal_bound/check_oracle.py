from pathlib import Path
import json,math,hashlib
P=Path(__file__).resolve().parent;d=json.loads((P/'VALIDATION_LOCK.json').read_text());checks=[]
def ck(n,e,t):checks.append(dict(name=n,error=e,gate=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
for i,s in enumerate(d['states']):
 for key in ['number','energy','thermal']:
  a=s[key];b=s['momentum_'+key];ck(f'Bessel versus scaledFD momentum {i} {key}',b/a-1,3e-11)
 ck('scaledpositivehole lies below analyticbound '+str(i),max(0.,s['log_hole_scaled_integral']-s['log_upper_hole_physical']),3e-12)
 ck('diluteFDcorrection negligible '+str(i),s['FD_relative_correction_bound'],1e-150)
 if i<8:
  expected_skip=i%4<2;ck('preregisteredhalfmin side '+str(i),0 if s['skip_bound']==expected_skip else 1,0);ck('preregisteredordinaryhole '+str(i),0 if (s['hole_float']==0)==expected_skip else 1,0)
out=dict(checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],candidate_imported=False,lock_sha256=hashlib.sha256((P/'VALIDATION_LOCK.json').read_bytes()).hexdigest());(P/'oracle_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out['passed'],out['failed'])
