"""Fresh post-lock references; unchanged preregistered gates, no retuning."""
from pathlib import Path
import importlib.util,sys,json,hashlib,time
P=Path(__file__).parent
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 if(P/'validation.json').exists():raise RuntimeError('Frozen validation exists')
 R=load('joint_validation_ref',P/'reference.py');A=load('joint_validation_api',P/'api.py');reg=json.loads((P/'PREREGISTRATION.json').read_text());lock=json.loads((P/'FITTING_LOCK.json').read_text())
 if lock['model_sha']!=sha(P/'model.json')or lock['preregistration_sha']!=sha(P/'PREREGISTRATION.json')or not lock['fresh_validation_file_absent']:raise RuntimeError('Prevalidation lock missing/mismatched')
 result=dict(lock_sha=sha(P/'FITTING_LOCK.json'),model_sha=lock['model_sha'],rows=[],checks=[],failed=[],started_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
 def ck(name,E,error,gate):result['checks'].append(dict(name=name,E=E,error=float(error),gate=gate,passed=abs(error)<=gate))
 for E in reg['fresh_heldout_MeV']:
  start=time.monotonic();c=R.detailed(E,R.COARSE);f=R.detailed(E,R.FINE);fit=A.evaluate(E,method='diagnostic_fit');a=c['total'];b=f['total']
  for key in['rate','transport_rate','outgoing_energy_moment']:
   ck('reference refinement '+key,E,a[key]/b[key]-1,1e-4);ck('raw locked fit '+key,E,fit[key]/b[key]-1,.01)
  ck('reference refinement signed_RT',E,(a['neutrino_energy_change']-b['neutrino_energy_change'])/(2*b['rate']),1e-4)
  delta=(fit['neutrino_energy_change']-b['neutrino_energy_change'])/(2*b['rate']);ck('raw locked fit signed_RT',E,delta,.005)
  if abs(b['neutrino_energy_change']/(2*b['rate']))>.25:ck('raw locked fit signed_relative',E,fit['neutrino_energy_change']/b['neutrino_energy_change']-1,.02)
  if abs(fit['g'])<=.08:
   auto=A.evaluate(E);ck('auto actual fallback flag',E,0 if auto['fallback']and auto['method']=='actual detailed CM quadrature'else 1,0)
   for key in R.KEYS:ck('fresh fallback matches reference '+key,E,(auto[key]-b[key])/(2*b['rate'])if key=='neutrino_energy_change'else auto[key]/b[key]-1,1e-12)
  else:auto=A.evaluate(E)
  result['rows'].append(dict(E=E,coarse=c,fine=f,raw_locked_fit=fit,auto=auto,seconds=time.monotonic()-start));result['failed']=[v for v in result['checks']if not v['passed']];(P/'validation_partial.json').write_text(json.dumps(result,indent=2));print(E,delta,'fallback',auto['fallback'],flush=True)
 result['model_unchanged']=sha(P/'model.json')==lock['model_sha'];result['passed']=sum(z['passed']for z in result['checks']);(P/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=result['passed'],failed=result['failed']),indent=2))
if __name__=='__main__':main()
