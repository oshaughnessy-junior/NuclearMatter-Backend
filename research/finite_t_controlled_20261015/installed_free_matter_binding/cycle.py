"""Prescribed mechanics only: independently compare RK4, entropy roots and path work."""
from pathlib import Path
import sys,json,math,time
import numpy as np
from scipy.integrate import simpson
from scipy.optimize import brentq
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P));import matter as M
assert json.loads((P/'partial_results.json').read_text())['status']=='PASS'
C=M.LOCK['cycle'];start=time.monotonic();steps=int(sys.argv[1]);assert steps in C['steps'];tau=C['tau_seconds'];dt=tau/steps;T=C['T0'];records=[]
def rhs(t,T):return M.compression_rhs(t,T,n0=C['n0'],Ye=C['Ye'],photons=C['photons'],tau=tau)
for i in range(steps+1):
 if time.monotonic()-start>112:raise TimeoutError('Cycle budget, partial checkpoint retained')
 t=i*dt;k1,s,ndot=rhs(t,T);records.append(dict(time=t,T=T,nB=s.nB,ndot=ndot,Tdot=k1,energy=s.energy,specific_energy=s.specific_energy,pressure=s.pressure,entropy_per_baryon=s.entropy_per_baryon,work_rate=s.pressure*ndot/s.nB**2))
 (P/f'cycle_{steps}_partial.json').write_text(json.dumps(dict(steps=steps,completed_steps=i,records=records),indent=2)+'\n')
 if i==steps:break
 k2=rhs(t+dt/2,T+dt*k1/2)[0];k3=rhs(t+dt/2,T+dt*k2/2)[0];k4=rhs(t+dt,T+dt*k3)[0];T+=dt*(k1+2*k2+2*k3+k4)/6
# Independent composite Simpson integral of pressure-work on saved endpoint states.
t=np.array([r['time']for r in records]);w=np.array([r['work_rate']for r in records]);work=float(simpson(w,x=t));abswork=float(simpson(abs(w),x=t));delta=records[-1]['specific_energy']-records[0]['specific_energy'];s0=records[0]['entropy_per_baryon'];checks=[]
def ck(name,value,gate):checks.append(dict(name=name,error=float(value),gate=gate,passed=bool(value<=gate)))
ck('entropy per baryon throughout',max(abs(r['entropy_per_baryon']/s0-1)for r in records),3e-6);ck('return temperature',abs(T/C['T0']-1),3e-6);ck('independent pressure work ledger scaled by total exchange',abs(delta-work)/abswork,3e-6)
quarters=[]
for j in range(4):
 lo=j*steps//4;hi=(j+1)*steps//4;qw=float(simpson(w[lo:hi+1],x=t[lo:hi+1]));du=records[hi]['specific_energy']-records[lo]['specific_energy'];quarters.append(dict(start=t[lo],end=t[hi],work=qw,delta_u=du,relative_ledger=abs(qw-du)/abs(qw)))
 # More sensitive local Simpson integration is recorded, not predeclared as global gate.
roots=[]
for tt in C['entropy_root_times']:
 n=C['n0']*math.exp(math.log(2)*math.sin(2*math.pi*tt/tau));tr=brentq(lambda temp:M.state(temp,n,C['Ye'],photons=C['photons']).entropy_per_baryon-s0,.75,3.5,xtol=1e-11,rtol=1e-13);idx=round(tt/dt);a=records[idx];ck('independent entropy root T at '+str(tt),abs(a['T']/tr-1),3e-6);roots.append(dict(time=tt,T_root=tr,T_RK4=a['T'],nB=n))
out=dict(status='PASS'if all(c['passed']for c in checks)else'FAIL',steps=steps,records=records,checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],simpson_work=work,total_absolute_work=abswork,delta_u=delta,quarter_work=quarters,entropy_roots=roots,seconds=time.monotonic()-start)
(P/f'cycle_{steps}.json').write_text(json.dumps(out,indent=2)+'\n');print(out['status'],out['passed'],out['failed'],'seconds',out['seconds'])
