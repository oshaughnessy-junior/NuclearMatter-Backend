from pathlib import Path
import json,copy,math,time
import numpy as np
import box as b
P=Path(__file__).resolve().parent;old=json.loads((P/'initial83pass_before_rate_guards.json').read_text());initial=old['initial'];g=b.grid(old['grid']['edges']);f=np.array(initial['f']);nB=old['nB'];r=old['physical_rate_evaluations'][0]
for field in ['j','chi','edges']:r[field]=np.array(r[field])
checks=[]
def reject(name,fn):
 try:fn()
 except ValueError:checks.append(dict(name=name,passed=True))
 else:checks.append(dict(name=name,passed=False))
for bad in [[0.,2.,1.],[1.,2.],[0.,math.nan]]:reject('bad grid'+str(bad),lambda bad=bad:b.grid(bad))
for arr in [np.full((2,4),math.nan),np.full((2,4),-.1),np.full((2,4),1.1),np.zeros((4,2))]:reject('bad f',lambda arr=arr:b.invariants(2.,.3,arr,nB,g))
for field,bad in [('T',2.1),('Ye',.31),('nB',.001),('edges',np.array([0,3,8,14,24])),('j',np.full((2,4),math.nan)),('chi',np.full((2,4),0.)),('j',np.full((2,4),-1.))]:
 rr=copy.deepcopy(r);rr[field]=bad
 reject('rate metadata/values '+field+str(bad),lambda rr=rr:b.step(2.,.3,f,nB,g,1e-10,rr))
for dt in [0.,-1.,math.nan]:reject('bad dt '+str(dt),lambda dt=dt:b.step(2.,.3,f,nB,g,dt,r))
# Cheap replay of one valid saved physical rate, same exact state/controls only.
row=old['runs'][0]['rows'][0];T,Y,F,_=b.step(2.,.3,f,nB,g,row['exchange']['dt'],r)
for name,error in [('T',abs(T-row['after']['T'])),('Ye',abs(Y-row['after']['Ye'])),('f',float(np.max(abs(F-np.array(row['after']['f'])))))]:checks.append(dict(name='guard repair valid replay '+name,passed=error<1e-13,error=error))
report=dict(passed=all(c['passed']for c in checks),checks=checks,scope='Input contracts plus one EXACT saved-state physical-rate replay; no changed-state cache reuse')
(P/'controls.json').write_text(json.dumps(report,indent=2)+'\n');print(len(checks),report['passed'])
