"""Finite closed spectral box from actual νe–e± redistribution kernels."""
from __future__ import annotations
import importlib.util,sys,math,time,hashlib
from pathlib import Path
from dataclasses import asdict
import numpy as np
from scipy.special import expit
from scipy.integrate import quad
from numpy.polynomial.legendre import leggauss
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True
FILES=['neutrino_lepton_scattering/scatter.py','leptonic_pair_annihilation/pair.py','relativistic_capture_oracle/events.py','thermodynamics/lepton_reference.py']
def pins():return {f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest()for f in FILES}
SOURCE_PINS=pins()
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m
S=load('box_scattering_pure','neutrino_lepton_scattering/scatter.py');L=load('box_pair_eos_pure','thermodynamics/lepton_reference.py')
T=2.;NE=3e-8;LEPTON=L.state_density(T,NE,S.ME);MU=LEPTON.chemical_potential;COUPLING=S.P.Couplings('e');HC=197.3269804

def bath():
 densities=[]
 for mu in [MU,-MU]:
  densities.append(quad(lambda p:p*p*expit((mu-math.hypot(p,S.ME))/T),0,250.,epsabs=1e-11,epsrel=2e-12)[0]/(math.pi**2*HC**3))
 res=max(abs(densities[0]/LEPTON.particle_density-1),abs(densities[1]/LEPTON.antiparticle_density-1),abs((densities[0]-densities[1])/NE-1))
 if res>2e-9:raise ArithmeticError('Pair bath forward gate')
 return dict(T=T,mu_e=MU,net_e_density=NE,particle_density=densities[0],antiparticle_density=densities[1],residual=res,eos=asdict(LEPTON),model='fixed pair-lepton thermostat; no material evolution')

def edge(E,F,*,angle=24,inner=24,tail=60.):
 if pins()!=SOURCE_PINS:raise RuntimeError('Frozen source changed')
 if not all(type(v)in(int,float)and math.isfinite(v)and 0<v<=50 for v in[E,F])or E==F:raise ValueError('Distinct positive finite selected energies<=50')
 if any(type(v)is not int or v<8 for v in[angle,inner]):raise ValueError('Integer quadrature orders>=8')
 if not math.isfinite(tail)or not 30<=tail<=90:raise ValueError('Explicit tail30..90')
 x,w=leggauss(angle);z=(x+1)/2;w=w/2;total=0.
 for zz,ww in zip(z,w):
  c=1-2*zz*zz
  for pos in[False,True]:total+=ww*4*zz*S.differential(T,MU,E,F,float(c),COUPLING,positron=pos,order=inner,phi=8,tail=tail)['value']
 if not math.isfinite(total)or total<=0:raise ArithmeticError('Unresolved positive edge')
 return float(total)

def grid(N,upper,**controls):
 if type(N)is not int or not 4<=N<=16 or not math.isfinite(upper)or not 12<=upper<=40:raise ValueError('Bounded4..16point grid; upper12..40')
 start=time.monotonic();x,w=leggauss(N);E=(x+1)*upper/2;w=w*upper/2;W=w*E*E/(2*math.pi**2*HC**3);K=np.zeros((N,N))
 for i in range(N):
  for j in range(N):
   if i!=j:
    if time.monotonic()-start>90:raise TimeoutError('Spectral edge experiment budget')
    K[i,j]=edge(float(E[i]),float(E[j]),**controls)
 A=W[:,None]*w[None,:]*K
 residual=max(abs(math.log(A[i,j]/A[j,i])-(E[i]-E[j])/T)for i in range(N)for j in range(i+1,N))
 return dict(E=E.tolist(),w=w.tolist(),W=W.tolist(),K=K.tolist(),A=A.tolist(),controls=controls,upper=upper,N=N,kms_log_residual=residual,seconds=time.monotonic()-start)

def collision(g,f):
 f=np.asarray(f,float);E=np.array(g['E']);W=np.array(g['W']);A=np.array(g['A']);N=len(E)
 if E.shape!=(N,)or W.shape!=(N,)or A.shape!=(N,N)or not all(np.all(np.isfinite(v))for v in[E,W,A])or np.any(E<=0)or np.any(W<=0)or np.any(A<0)or np.any(np.diag(A)!=0)or any(A[i,j]<=0 for i in range(N)for j in range(N)if i!=j):raise ValueError('Finite positive spectral measures/edges and exact zero self-transitions required')
 if f.shape!=(N,)or not np.all(np.isfinite(f))or np.any(f<=0)or np.any(f>=1):raise ValueError('Finite strictly interior occupation array required')
 out=np.zeros(N);rad=0.;entropy=0.;events=0.
 for i in range(N):
  for j in range(i+1,N):
   lf=math.log(A[i,j])+math.log(f[i])+math.log1p(-f[j]);lb=math.log(A[j,i])+math.log(f[j])+math.log1p(-f[i]);d=lf-lb;J=math.copysign(math.exp(max(lf,lb))*(-math.expm1(-abs(d))),d)
   out[i]-=J;out[j]+=J;rad+=(E[j]-E[i])*J;entropy+=J*d;events+=math.exp(lf)+math.exp(lb)
 # Independent matrix-row assembly, not antisymmetry-enforced number zero.
 F=A*f[:,None]*(1-f[None,:]);independent=np.sum(F,axis=0)-np.sum(F,axis=1)
 derivative=out/W;sr=-float(np.dot(np.log(f/(1-f)),out));macro=sr-rad/T
 return dict(df=derivative.tolist(),dn=out.tolist(),radiation_energy=rad,matter_energy=-rad,entropy_production=entropy,macro_entropy=macro,number=sum(out),activity=events,independent_dn=independent.tolist())

def evolve(g,Tr,mu,*,duration=.08,steps=16):
 if not all(type(v)in(int,float)and math.isfinite(v)for v in[Tr,mu,duration])or Tr<=0 or duration<=0 or type(steps)is not int or steps<1:raise ValueError('Finite positive radiation/time and integer steps required')
 E=np.array(g['E']);W=np.array(g['W']);f=expit((mu-E)/Tr);initial=f.copy();dt=duration/steps;heat=0.;prod=0.;minf=float(min(f));maxf=float(max(f));rows=[]
 def h(v):return -np.dot(W,v*np.log(v)+(1-v)*np.log1p(-v))
 for k in range(steps):
  a=collision(g,f);dn=np.array(a['df']);pred=f+dt*dn
  if np.any(pred<=0)or np.any(pred>=1):raise ArithmeticError('Positivity stage failed; refine time')
  b=collision(g,pred);new=f+dt/2*(dn+np.array(b['df']))
  if np.any(new<=0)or np.any(new>=1):raise ArithmeticError('Positivity update failed')
  dh=dt/2*(a['matter_energy']+b['matter_energy']);heat+=dh;prod+=dt/2*(a['entropy_production']+b['entropy_production']);f=new;minf=min(minf,float(min(f)));maxf=max(maxf,float(max(f)));rows.append(dict(t=(k+1)*dt,number=float(W@f),radiation_energy=float((W*E)@f),matter_heat=heat))
 return dict(Tr=Tr,mu=mu,duration=duration,steps=steps,initial=initial.tolist(),final=f.tolist(),number_initial=float(W@initial),number_final=float(W@f),radiation_energy_initial=float((W*E)@initial),radiation_energy_final=float((W*E)@f),matter_heat=heat,number_ledger=float(W@(f-initial)),energy_ledger=float((W*E)@(f-initial))+heat,entropy_radiation=float(h(f)-h(initial)),entropy_bath=heat/T,entropy_production_integrated=prod,min_occupation=minf,max_occupation=maxf,history=rows)
