from pathlib import Path
import json,sys,importlib.util,time
import numpy as np
p=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('transparent_final_controls',p/'balance.py');m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
def run():
 start=time.monotonic();d=json.loads((p/'results.json').read_text());r=d['balanced'];checks=[]
 def check(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(abs(e)<=t)))
 for x in [float('nan'),float('inf'),1,0,'fine',None]:
  try:m.evaluate(.3,refined=x);ok=False
  except ValueError:ok=True
  check('literalbool '+repr(x),0 if ok else 1,0)
 old=(p/'experiment_balance_source.py.txt').read_text();new=(p/'balance.py').read_text();check('only added literal control guard after experiment load',0 if new.replace(" if not isinstance(refined,bool):raise ValueError('Literal refined bool required')\n",'')==old else 1,0)
 fresh=m.evaluate(r['Ye'],True)
 for c in r['channels']:
  for k,v in r['channels'][c].items():check('fresh final source '+c+' '+k,fresh['channels'][c][k]/v-1,0)
 bath,state=m.material(r['Ye']);b=vars(bath)
 ec=m.EC.integrate(bath,proton_order=28,electron_order=48,initial_order=12,outgoing_order=12,phi_order=12,tail=80.,scenarios=[dict(name='empty',mode='empty')])
 pc=m.PC.integral(b,order_n=32,order_e=52,order_angle=12,tail=80.)
 beta=m.BETA.integrate(m.BETA.K.Bath(**b),rest_order=24,parent_order=28,orient_order=10,phi_order=16,coordinate='electron',parent_coordinate='direct_p',tail=100.,scenarios=[dict(name='empty',mode='empty')])
 tails={'EC':ec['scenarios']['empty']['net'],'PC':dict(zip(m.PC.NAMES,np.asarray(pc['sums'])[0,0].tolist())),'beta':beta['scenarios']['empty']['net']}
 for c in tails:
  for k,v in tails[c].items():check('isolated tail '+c+' '+k,v/r['channels'][c][k]-1,3e-6)
 out=dict(checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],fresh_final_source=fresh,tail_sources=dict(EC=ec,PC=pc,beta=beta),seconds=time.monotonic()-start)
 (p/'controls_tail.json').write_text(json.dumps(out,indent=2)+'\n');print(dict(passed=out['passed'],failed=out['failed'],seconds=out['seconds']))
if __name__=='__main__':run()
