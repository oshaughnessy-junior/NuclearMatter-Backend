"""Kinematics/KMS/normalization witnesses; NOT a finite-T tensor kernel."""
from pathlib import Path
import sys,math,json,time,hashlib
import numpy as np
from scipy.special import roots_genlaguerre,log_expit
ROOT=Path(__file__).resolve().parent.parent;HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'muon_full_current'))
import current as F
V=F.V;start=time.monotonic();checks=[];details={}
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and abs(error)<=tol)))

# q-axis angle, exact delta Jacobian and support polynomial from dispersions.
for ma,mb,p,q,x in[(939.,938.,100.,50.,.35),(700.,710.,180.,90.,-.7),(939.,938.,40.,.5,.9)]:
 Ea=math.hypot(ma,p);Eb=math.sqrt(mb*mb+p*p+q*q+2*p*q*x);w=Eb-Ea
 C=ma*ma-mb*mb+w*w-q*q
 reconstructed=(2*w*Ea+C)/(2*p*q)
 gate(f'qaxis root {ma}{p}{q}',reconstructed-x,3e-10)
 h=1e-5
 def Ebfun(c):return math.sqrt(mb*mb+p*p+q*q+2*p*q*c)
 jacnum=(Ebfun(x+h)-Ebfun(x-h))/(2*h)
 gate(f'energy delta Jacobian {ma}{p}{q}',jacnum/(p*q/Eb)-1,3e-7)
 poly=(q*q-w*w)*Ea*Ea-w*C*Ea-(C*C/4+q*q*ma*ma)
 expected=p*p*q*q*(1-x*x)
 gate(f'support polynomial {ma}{p}{q}',poly/expected-1,2e-9)

# Pointwise KMS including vector shifts; mu's are total physical potentials.
for Ua,Ub in[(0.,0.),(12.,-4.)]:
 for E,El,mul in[(40.,50.,24.),(150.,120.,95.),(70.,100.,-35.)]:
  T=4.;Ea=960.+Ua;omega=E-El;Eb=Ea+omega;mua,mub=950.,935.
  logfa=log_expit((mua-Ea)/T);logva=log_expit((Ea-mua)/T)
  logfb=log_expit((mub-Eb)/T);logvb=log_expit((Eb-mub)/T)
  logfl=log_expit((mul-El)/T);logvl=log_expit((El-mul)/T)
  ratio=float(logfb+logfl+logva-logfa-logvb-logvl)
  gate(f'occupation KMS U{Ua}{Ub}E{E}',ratio+(E-(mul+mub-mua))/T,3e-13)

# Full spin trace, divided by relativistic baryon state normalizations -> spin2.
for M in(1e5,1e6):
 a=np.array([math.hypot(M,100.),0.,0.,100.]);b=np.array([math.sqrt(M*M+100.**2+50.**2),50.,0.,100.])
 trace=np.trace((V.slash(b)+M*V.I)@V.G[0]@(V.slash(a)+M*V.I)@V.G[0]).real
 ratio=trace/(4*a[0]*b[0])
 gate(f'NR spin2 density normalization M{M}',ratio/2-1,2e-7 if M==1e5 else 2e-9)

# Maxwell-Juttner dilute gas: thermal shape is exp[-(Ea-ma)/T].
# One integrates n_target <v_Moller sigma(s)>; occupation amplitude cancels.
shells,weights=roots_genlaguerre(10,.5);cs,cws=np.polynomial.legendre.leggauss(10)
rows=[]
for anti in(False,True):
 ch=V.Channel(V.MP if anti else V.MN,V.MN if anti else V.MP,V.MMU,anti)
 static=F.total(ch,150.,order=24);errors=[]
 for T in(.1,.01,.001):
  radial=[];values=[];M=ch.initial_mass
  for x,weight in zip(shells,weights):
   Ea=M+T*x;p=math.sqrt(T*x*(T*x+2*M))
   radialweight=weight*math.sqrt(T*(T*x+2*M))*Ea*T
   average=0.
   for cosine,cw in zip(cs,cws):
    relv=1-p/Ea*cosine;Erest=150.*Ea/M*relv
    average+=cw*.5*relv*F.total(ch,float(Erest),order=12)
   radial.append(radialweight);values.append(radialweight*average)
  mean=float(sum(values)/sum(radial));error=mean/static-1;errors.append(abs(error))
  n=1e-18;fugacity=n*math.pi**2*V.HBARC**3/float(sum(radial))
  rate=n*1e39*2.99792458e10*mean
  rows.append(dict(antineutrino=anti,T=T,n_fm3=n,fugacity=fugacity,mean_vsigma_cm2=mean,stationary_sigma_cm2=static,relative_difference=error,chi_s_inv=rate))
  gate(f'dilute occupation {anti}T{T}',max(0.,fugacity-1e-8),0.)
 gate(f'cold dilute target convergence {anti}',0. if errors[2]<errors[1]/8 and errors[1]<errors[0]/8 else 1.,0.)
 gate(f'cold dilute limit {anti}',errors[-1],3e-5)
details['dilute_moving_targets']=rows
refinements=[]
for anti in(False,True):
 ch=V.Channel(V.MP if anti else V.MN,V.MN if anti else V.MP,V.MMU,anti);M=ch.initial_mass;T=.1
 rx,rw=roots_genlaguerre(16,.5);cx,cw=np.polynomial.legendre.leggauss(16);den=num=0.
 for x,weight in zip(rx,rw):
  Ea=M+T*x;p=math.sqrt(T*x*(T*x+2*M));rweight=weight*math.sqrt(T*(T*x+2*M))*Ea*T;average=0.
  for cosine,wc in zip(cx,cw):
   speed=1-p/Ea*cosine;average+=wc*.5*speed*F.total(ch,float(150.*Ea/M*speed),order=16)
  den+=rweight;num+=rweight*average
 coarse=next(r['mean_vsigma_cm2']for r in rows if r['antineutrino']==anti and r['T']==T)
 refined=num/den
 gate(f'dilute radial angular sigma refinement {anti}',refined/coarse-1,3e-9)
 refinements.append(dict(antineutrino=anti,T=T,coarse=coarse,refined=refined,relative_difference=refined/coarse-1,radial_order=16,angle_order=16,sigma_order=16))
details['dilute_quadrature_refinement']=refinements

# Algebraic prefactor closure to the frozen NR optical kernel.
sys.path.insert(0,str(ROOT/'allowed_opacity'));import allowed_capture
K=V.GF**2*V.VUD**2*V.HBARC**3/(4*math.pi**2*6.582119569e-22)
gate('NR optical prefactor closure',K/allowed_capture.K-1,3e-15)
out=dict(passed=all(c['passed']for c in checks),checks=checks,details=details,runtime_seconds=time.monotonic()-start,scope='kinematic and normalization contract witnesses, not an implemented finiteT tensor',source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in[Path(__file__),ROOT/'muon_full_current/current.py',ROOT/'muon_vacuum/vacuum_cc.py',ROOT/'allowed_opacity/allowed_capture.py']})
(HERE/'results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(passed=out['passed'],checks=len(checks),seconds=out['runtime_seconds'],failed=[x for x in checks if not x['passed']])))
if not out['passed']:raise SystemExit(1)
