"""Independent phase-volume normalization and relativistic support review."""
from pathlib import Path
import math,json,time,hashlib,sys
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
from scipy.special import log_expit
P=Path(__file__).resolve().parent;ROOT=P.parent;sys.dont_write_bytecode=True
paths=list((ROOT/'relativistic_response_contract').iterdir());paths=[p for p in paths if p.is_file()]
sha=lambda:{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
before=sha();start=time.monotonic();checks=[];rows=[]
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error) and error<tol)))
# Integrate lab final-state phase volume by its radial lepton delta root.
for ma,mb,ml,E,pa,ct in [(939.565,938.272,105.658,173.,0.,.3),(938.272,939.565,.511,29.,137.,-.43),(700.,710.,105.658,230.,190.,.67)]:
 Ea=math.hypot(ma,pa);Et=Ea+E;Ptot=math.sqrt(E*E+pa*pa+2*E*pa*ct);s=Et*Et-Ptot*Ptot;root=math.sqrt(s);lam=(s-(mb+ml)**2)*(s-(mb-ml)**2);pstar=math.sqrt(lam)/(2*root);expected=pstar/(4*math.pi*root)
 def angular(c):
  def residual(p):return Et-math.hypot(ml,p)-math.sqrt(mb*mb+Ptot*Ptot+p*p-2*Ptot*p*c)
  if residual(0)<=0:raise ValueError('Witness chosen outside simple one-radial-root regime')
  p=brentq(residual,0.,Et,xtol=2e-12);El=math.hypot(ml,p);Eb=Et-El;derivative=p/El+(p-Ptot*c)/Eb
  return p*p/(8*math.pi*El*Eb*abs(derivative))
 measured,err=quad(angular,-1.,1.,epsabs=1e-13,epsrel=2e-11)
 gate('independent lab/CM phase volume '+str((ma,ml,E)),abs(measured/expected-1),3e-10)
 dot=E*(Ea-pa*ct);velocity=dot/(E*Ea);sigma=expected/(4*dot)
 # Per-spin fa but physical target n=2I: explicit2 with initial averaged matrix.
 rate_per_n=(2/(2*E)/(2*Ea))*measured/2
 gate('absolute spin2 dilute c-sigma closure '+str((ma,ml,E)),abs(rate_per_n/(velocity*sigma)-1),3e-10)
 rows.append(dict(kind='phase_volume',ma=ma,mb=mb,ml=ml,Enu=E,target_p=pa,target_cos=ct,s=s,CM_phase_volume=expected,lab_phase_volume=measured,Moller_v=velocity,sigma_for_unit_target_averaged_amplitude=sigma,opacity_per_target_density=rate_per_n))
# Gaussian energy delta independently compared against reduced q-axis measure.
for ma,mb,p,q,x in [(939.,938.,132.,74.,-.37),(700.,710.,190.,60.,.53)]:
 Ea=math.hypot(ma,p);Eb=math.sqrt(mb*mb+p*p+q*q+2*p*q*x);w=Eb-Ea;expected=1/(4*Ea*p*q) # int dx delta/(4EaEb)
 vals=[]
 for eps in (.03,.01,.003):
  def integrand(y):
   Eby=math.sqrt(mb*mb+p*p+q*q+2*p*q*y);d=(w+Ea-Eby)/eps
   return math.exp(-d*d/2)/(math.sqrt(2*math.pi)*eps*4*Ea*Eby)
  width=eps*Eb/(p*q);lo=max(-1.,x-12*width);hi=min(1.,x+12*width);value=quad(integrand,lo,hi,epsabs=1e-19,epsrel=1e-10,points=[x])[0];vals.append(value)
  gate('Gaussian original-angle delta '+str((ma,eps)),abs(value/expected-1),2e-10)
 rows.append(dict(kind='delta_jacobian',ma=ma,mb=mb,p=p,q=q,x=x,w=w,expected_original_angle_integral=expected,gaussian_delta_values=vals))
# Sample exact mass-shell kinematics across spacelike, timelike and lightlike support.
rng=np.random.default_rng(20261015)
for ma,mb,q in [(939.,938.,54.),(700.,810.,20.),(810.,700.,20.)]:
 for p in (30.,140.,370.):
  for x in (-.8,.21,.93):
   Ea=math.hypot(ma,p);Eb=math.sqrt(mb*mb+p*p+q*q+2*p*q*x);w=Eb-Ea;C=ma*ma-mb*mb+w*w-q*q
   poly=(q*q-w*w)*Ea*Ea-w*C*Ea-(C*C/4+q*q*ma*ma)
   expected=p*p*q*q*(1-x*x)
   gate('support invariant '+str((ma,mb,p,x)),abs(poly/expected-1),5e-9)
   xx=(C+2*w*Ea)/(2*p*q);gate('support angle reconstruction '+str((ma,mb,p,x)),abs(xx-x),3e-10)
   gate('original energy shell '+str((ma,mb,p,x)),abs(Ea+w-math.sqrt(mb*mb+p*p+q*q+2*p*q*xx)),3e-10)
# Lightlike kinetic transfer has a LINEAR threshold for unequal masses.
for ma,mb,w,q in [(810.,700.,-30.,30.),(700.,810.,30.,30.)]:
 C=ma*ma-mb*mb;threshold=-(C*C/4+q*q*ma*ma)/(w*C)
 # Select energies on both sides; exact original angle defines support.
 for E in (max(ma,mb-w,threshold)*1.1,max(ma,mb-w,threshold)*.9):
  if E<=ma or E+w<=mb:continue
  p=math.sqrt(E*E-ma*ma);x=(C+2*w*E)/(2*p*q);lhs=-w*C*E-(C*C/4+q*q*ma*ma)
  gate('linear support decision '+str((ma,E)),float((lhs>=0)!=(abs(x)<=1)),.1)
# Pointwise total-potential KMS; shifts must not be counted twice.
for Ua,Ub in [(0.,0.),(31.,-12.)]:
 for T,Ea,w,mua,mub,mul,El in [(3.7,970.,-11.,960.,925.,42.,74.),(1.3,750.,82.,740.,815.,-27.,121.)]:
  ea=Ea+Ua;eb=Ea+w+Ub;omega=eb-ea;Enu=omega+El
  la,va=log_expit((mua-ea)/T),log_expit((ea-mua)/T);lb,vb=log_expit((mub-eb)/T),log_expit((eb-mub)/T);ll,vl=log_expit((mul-El)/T),log_expit((El-mul)/T)
  ratio=float(lb+va+ll-la-vb-vl);expected=-(Enu-(mul+mub-mua))/T
  gate('KMS total-potential '+str((Ua,T)),abs(ratio-expected),5e-13)
# Heavy-mass full-trace spin2 and lepton contraction close all numerical prefactors.
sys.path.insert(0,str(ROOT/'allowed_opacity'));import allowed_capture as nr
for M in (1e5,1e7):
 a=np.array([math.hypot(M,131.),131.,0.,0.]);b=np.array([math.hypot(M,187.),0.,187.,0.]);dot=a[0]*b[0]-np.dot(a[1:],b[1:]);T00=4*(2*a[0]*b[0]-dot+M*M)
 gate('heavy trace spin2 '+str(M),abs(T00/(4*a[0]*b[0])/2-1),2e-6)
for Enu,El,ml,c in [(41.,37.,.511,-.47),(170.,131.,105.658,.68)]:
 pl=math.sqrt(El*El-ml*ml);L00=8*Enu*(El+pl*c);Lii=8*Enu*(3*El-pl*c);contraction=L00+nr.GA**2*Lii
 tensor_pref=nr.GF**2*nr.VUD**2/(32*math.pi**2)*pl/Enu*contraction
 direct_pref=nr.GF**2*nr.VUD**2/(4*math.pi**2)*El*pl*((1+pl/El*c)+nr.GA**2*(3-pl/El*c))
 gate('NR allowed contraction '+str(ml),abs(tensor_pref/direct_pref-1),3e-15)
 restored=nr.GF**2*nr.VUD**2*nr.HBARC**3/(4*math.pi**2*6.582119569e-22)
 gate('NR hbar hbarc optical units '+str(ml),abs(restored/nr.K-1),3e-15)
gate('all author source/results unchanged',float(sha()!=before),.1)
out=dict(passed=all(c['passed']for c in checks),checks=checks,rows=rows,source_hashes_before=before,source_hashes_after=sha(),runtime_seconds=time.monotonic()-start)
(P/'results.json').write_text(json.dumps(out,indent=2)+'\n');print(sum(c['passed']for c in checks),'/',len(checks),'runtime',out['runtime_seconds'])
for c in checks:
 if not c['passed']:print('FAIL',c)
