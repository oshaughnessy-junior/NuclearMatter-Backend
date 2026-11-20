"""Bounded actual changing-state integration and forward/control qualifications."""
from __future__ import annotations
from pathlib import Path
import json,time,math,hashlib
import numpy as np
import element as E
HERE=Path(__file__).resolve().parent
checks=[]
def check(name,error,limit):
 error=float(error);checks.append(dict(name=name,error=error,limit=limit,passed=bool(math.isfinite(error)and error<=limit)))
def dump(name,obj):
 (HERE/name).write_text(json.dumps(obj,indent=2,default=lambda x:x.item()if isinstance(x,np.generic)else str(x)))
def main():
 start=time.monotonic();src=E.Sources();runs=[]
 for steps in [1,2,4]:
  runs.append(E.trajectory(steps,src));dump('partial_trajectories.json',dict(runs=runs,evaluations=src.evaluations,identity=src.pin))
 a=runs[0]['initial'];z=src.get(a)
 check('initial root-joined number forward',abs(z['R']/1.117361868663e-6-1),2e-8)
 check('initial root-joined energy forward',abs(z['Q']/1.199463580173e-5-1),2e-8)
 fine=[]
 for tag,s in [('initial',a),('changed_endpoint',runs[-1]['final'])]:
  coarse=src.get(s);ref=src.get(s,True);fine.append(dict(tag=tag,coarse=coarse,fine=ref))
  for key in ['R','Q','pc_R','pc_Q','beta_R','beta_Q']:
   check(tag+' actual source refinement '+key,abs(coarse[key]/ref[key]-1),1e-7)
  for h in [1e-3,5e-4]:
   dt=(E.state(s['T']+h,s['Ye'])['u']-E.state(s['T']-h,s['Ye'])['u'])/(2*h)
   dy=(E.state(s['T'],s['Ye']+h)['u']-E.state(s['T'],s['Ye']-h)['u'])/(2*h)
   check(tag+' genuine FD heat derivative '+str(h),abs(dt/s['u_T']-1),1e-6)
   check(tag+' genuine FD composition derivative '+str(h),abs(dy/s['u_Y']-1),1e-6)
 for run in runs:
  check(str(run['steps'])+' energy escaped ledger',abs(run['energy_ledger_relative_exchange']),1e-8)
  check(str(run['steps'])+' material plus escaped lepton ledger',abs(run['lepton_ledger'])/run['escaped']['number'],1e-12)
  check(str(run['steps'])+' channel number decomposition',abs((run['escaped']['pc_number']+run['escaped']['beta_number'])/run['escaped']['number']-1),1e-14)
  check(str(run['steps'])+' channel energy decomposition',abs((run['escaped']['pc_energy']+run['escaped']['beta_energy'])/run['escaped']['energy']-1),1e-14)
  for rec in run['records']:
   # Forward separately before any derivative/time-accuracy conclusion.
   s=rec['state'];check(str(run['steps'])+' forward density '+str(rec['index']),s['density_residual'],2e-9)
  check(str(run['steps'])+' energy inversion',abs(E.state(run['final']['T'],run['final']['Ye'])['u']-run['final']['u']),2e-9)
 changes={}
 for k in ['T','Ye','u']:
  vals=[r['final'][k]for r in runs];d12=abs(vals[1]-vals[0]);d24=abs(vals[2]-vals[1]);order=math.log2(d12/d24)
  changes[k]=dict(endpoints=vals,change=vals[-1]-a[k],difference12=d12,difference24=d24,observed_order=order,first_order_richardson_remaining=d24)
  check(k+' measured first-order refinement',abs(order-1),.12)
 check('nonvacuous composition change',.02/abs(changes['Ye']['change']),1.)
 check('distinct evaluated changed states',float(len(src.evaluations)<7),0.)
 # Invalid inputs and source identity are tested without any expensive quadrature.
 for name,fn in [('negativeT',lambda:E.state(-1,.3)),('badYe',lambda:E.state(2,1.)),('nanenergy',lambda:E.invert(float('nan'),.3)),('zero steps',lambda:E.trajectory(0,src)),('nonfinite duration',lambda:E.trajectory(1,src,float('nan')))]:
  try:fn();rejected=False
  except (ValueError,ArithmeticError):rejected=True
  check('reject '+name,float(not rejected),0.)
 saved=src.pin;src.pin={}
 try:src.get(a);rejected=False
 except RuntimeError:rejected=True
 src.pin=saved;check('reject stale source identity before reuse',float(not rejected),0.)
 result=dict(checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],runs=runs,source_refinement=fine,time_refinement=changes,source_evaluations=src.evaluations,identity=src.pin,seconds=time.monotonic()-start,scope='ONLY n beta decay + positron capture, empty radiation, fixed nB, particle-only free baryons and pair electrons; no full neutrino inventory')
 dump('results.json',result);print(json.dumps(dict(passed=result['passed'],failed=result['failed'],seconds=result['seconds'],changes=changes),default=float,indent=2))
if __name__=='__main__':main()
