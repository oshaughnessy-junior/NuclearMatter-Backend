"""Independent momentum EOS and Simpson pressure-work review, weak off only."""
from pathlib import Path
import sys,json,math,time,hashlib,atexit
import numpy as np
from scipy.special import expit
from scipy.integrate import quad,simpson
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent;sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'fd_prescribed_density'))
import density_element as a
start=time.monotonic();checks=[];raw={};phase='start'
paths=[ROOT/'fd_prescribed_density'/s for s in ['density_element.py','weak_off.py','weak_off_results.json','initial_work_budget_stop.txt']]
hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths}
def save():
 end={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths}
 (HERE/'results.json').write_text(json.dumps(dict(passed=all(c['passed']for c in checks)and phase=='complete'and hashes==end,phase=phase,checks=checks,raw=raw,runtime_seconds=time.monotonic()-start,source_sha256=hashes,end_source_sha256=end,scope='Selected momentum EOS density derivatives, independent entropy roots, Simpson pressure work; weakOFF only'),indent=2)+'\n')
atexit.register(save)
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and error<tol)))
cache={}
def direct(T,nB,Y):
 if time.monotonic()-start>43:raise TimeoutError('Review45s numerical bound; earlier probes checkpointed')
 key=(T,nB,Y)
 if key in cache:return cache[key]
 s=a.state(T,nB,Y);m=s['matter'];hc=a.element.physics.HBARC;U=P=S=0.;densities=[]
 for sp in [m.neutron,m.proton]:
  M=sp.mass;mu=sp.kinetic_mu;end=math.sqrt(2*M*(max(mu,0)+100*T));pref=1/(math.pi**2*hc**3)
  f=lambda p:expit((mu-p*p/(2*M))/T)
  density=quad(lambda p:p*p*f(p),0,end,epsabs=1e-11,epsrel=3e-12)[0]*pref
  K=quad(lambda p:p**4/(2*M)*f(p),0,end,epsabs=1e-10,epsrel=3e-12)[0]*pref
  def entropy(p):
   z=abs((p*p/(2*M)-mu)/T)
   return p*p*(np.logaddexp(0.,-z)+z*expit(-z))
  S+=quad(entropy,0,end,epsabs=1e-10,epsrel=3e-12)[0]*pref;U+=K+sp.offset*density;P+=2*K/3;densities.append(density)
 el=m.electron;me=el.mass;mu=el.chemical_potential;end=max(abs(mu),me)+100*T;pref=1/(math.pi**2*hc**3)
 energy=lambda p:math.hypot(me,p)
 occ=lambda p,sign:expit((sign*mu-energy(p))/T)
 ne=quad(lambda p:p*p*(occ(p,1)-occ(p,-1)),0,end,epsabs=1e-11,epsrel=3e-12)[0]*pref
 U+=quad(lambda p:p*p*energy(p)*(occ(p,1)+occ(p,-1)),0,end,epsabs=1e-10,epsrel=3e-12)[0]*pref
 P+=quad(lambda p:p**4/energy(p)*(occ(p,1)+occ(p,-1)),0,end,epsabs=1e-10,epsrel=3e-12)[0]*pref/3
 def entropy_e(p):
  zz=[abs((energy(p)-sign*mu)/T)for sign in [1,-1]]
  return p*p*sum(np.logaddexp(0.,-z)+z*expit(-z)for z in zz)
 S+=quad(entropy_e,0,end,epsabs=1e-10,epsrel=3e-12)[0]*pref
 result=dict(u=U/nB,P=P,s=S/nB,densities=[*densities,ne]);cache[key]=result;return result
phase='heldout_derivatives';T,n,Y=2.15,1.03e-4,.287;s=a.state(T,n,Y);central=direct(T,n,Y)
gate('heldout forward independent densities',np.max(abs(np.array(central['densities'])/[n*(1-Y),n*Y,n*Y]-1)),2e-10)
gate('heldout forward u',abs(central['u']/s['u_MeV_per_baryon']-1),2e-10)
gate('heldout forward P',abs(central['P']/s['P_MeV_fm3']-1),2e-10)
gate('heldout forward entropy',abs(central['s']/s['entropy_per_baryon']-1),2e-10)
for h in [2e-4,1e-4]:
 plus,minus=direct(T,n*(1+h),Y),direct(T,n*(1-h),Y)
 un=(plus['u']-minus['u'])/(2*n*h);sn=(plus['s']-minus['s'])/(2*n*h)
 gate(f'heldout independent u_n h{h}',abs(un/s['u_n_MeV_fm3']-1),3e-7)
 gate(f'heldout independent entropy_n h{h}',abs(sn/(-s['dP_dT_fixed_nY']/n**2)-1),3e-7)
raw['heldout']=dict(T=T,nB=n,Ye=Y,direct=central,analytic_u_n=s['u_n_MeV_fm3']);save()
phase='entropy_root';T0,n0,Y=2.,1e-4,.3;n1=n0*math.exp(.02);initial=direct(T0,n0,Y);S0=initial['s']
def temperature(n):return brentq(lambda T:direct(T,n,Y)['s']-S0,1.95,2.08,xtol=2e-12,rtol=2e-13)
T1=temperature(n1);end=direct(T1,n1,Y);author=json.loads((ROOT/'fd_prescribed_density/weak_off_results.json').read_text())
gate('independent entropy root vs author peak',abs(T1/author['compression']['T']-1),2e-10)
gate('independent endpoint entropy',abs(end['s']/S0-1),2e-11)
raw['entropy_root']=dict(T=T1,nB=n1,s=end['s'],target=S0,u=end['u']);save()
phase='pressure_work';work=[]
for intervals in [4,8]:
 ns=np.linspace(n0,n1,intervals+1);values=[]
 for nn in ns:
  temp=temperature(float(nn));values.append(direct(temp,float(nn),Y)['P']/nn**2)
 value=float(simpson(values,x=ns));work.append(value)
 raw[f'Simpson{intervals}']=dict(nB=ns.tolist(),P_over_n2=values,work=value);save()
gate('independent Simpson4->8 pressure work',abs(work[0]/work[1]-1),2e-9)
gate('independent pressure work vs independent energy',abs(work[1]/(end['u']-initial['u'])-1),2e-9)
gate('independent pressure work vs authorGauss8',abs(work[1]/author['compression']['work_pressure_integral']-1),2e-9)
phase='complete';save();print(json.dumps(dict(passed=all(c['passed']for c in checks),checks=len(checks),runtime=time.monotonic()-start,failed=[c for c in checks if not c['passed']]),indent=2))
