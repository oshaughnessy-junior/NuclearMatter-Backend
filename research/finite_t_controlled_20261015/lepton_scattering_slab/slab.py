"""Finite-energy S_N slab, physical inelastic e± scattering, fixed pair bath.

Axisymmetric in azimuth; upwind cell populations. No hydro/material evolution.
"""
from __future__ import annotations
import sys,importlib.util,hashlib,math,time
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.integrate import solve_ivp
from numpy.polynomial.legendre import leggauss
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True
FILES=['neutrino_lepton_scattering/scatter.py','leptonic_pair_annihilation/pair.py','relativistic_capture_oracle/events.py','thermodynamics/lepton_reference.py']
def identity():return{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest()for f in FILES}
PINS=identity()
def load(name,file):
 sp=importlib.util.spec_from_file_location(name,ROOT/file);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
S=load('slab_physical_scattering','neutrino_lepton_scattering/scatter.py');L=load('slab_pair_eos','thermodynamics/lepton_reference.py');T=2.;NE=3e-8;LEPTON=L.state_density(T,NE,S.ME);MU=LEPTON.chemical_potential;C=S.P.Couplings('e');HC=197.3269804;C_CM_S=2.99792458e10

def _finite_real(value):
 return type(value)in(int,float)and math.isfinite(value)

def _real_array(value,name):
 try:raw=np.asarray(value)
 except (TypeError,ValueError)as ex:raise ValueError('Real numeric '+name+' required')from ex
 if raw.dtype.kind not in 'iuf':raise ValueError('Real numeric '+name+' required')
 out=np.asarray(raw,float)
 if not np.all(np.isfinite(out)):raise ValueError('Finite '+name+' required')
 return out

def _phase_arrays(g):
 E=_real_array(g['E'],'energy nodes');W=_real_array(g['W'],'phase weights');A=_real_array(g['A'],'physical transitions');N=len(E)if E.ndim==1 else 0
 m=_real_array(g['mu'],'angular nodes')
 if not N or W.shape!=(N,)or m.shape!=(N,)or A.shape!=(N,N)or np.any(E<=0)or np.any(W<=0)or np.any(abs(m)>=1)or np.any(A<0)or any(A[i,j]<=0 for i in range(N)for j in range(N)if i!=j):raise ValueError('Positive finite spectral measures/transitions and interior angular nodes required')
 return E,W,A

def _occupation(value,shape,name):
 out=_real_array(value,name)
 if out.shape!=shape or np.any(out<=0)or np.any(out>=1):raise ValueError('Strictly interior '+name+' of matching shape required')
 return out

def angular_edge(E,F,mu,mup,*,azimuth=32,inner=32,tail=80.):
 """Mean over relative azimuth of D; outgoing measure is wE*wmu*Dbar.

Coincident (E,mu) is excluded before integration: axisymmetric f cancels
its exact forward/reverse flux identically; no activity for it is claimed.
 """
 if identity()!=PINS:raise RuntimeError('Physical source changed')
 if not all(type(z)in(int,float)and math.isfinite(z)for z in[E,F,mu,mup,tail])or min(E,F)<=0 or max(E,F)>50 or abs(mu)>=1 or abs(mup)>=1 or E==F and mu==mup:raise ValueError('Distinct finite phase modes, E<=50,interior mu required')
 if type(azimuth)is not int or azimuth<8 or type(inner)is not int or inner<8 or not 30<=tail<=90:raise ValueError('Bounded explicit integer rules and tail')
 total=0.
 for a in range(azimuth):
  phi=2*math.pi*(a+.5)/azimuth;co=mu*mup+math.sqrt((1-mu*mu)*(1-mup*mup))*math.cos(phi)
  for pos in[False,True]:total+=S.differential(T,MU,E,F,co,C,positron=pos,order=inner,phi=8,tail=tail)['value']/azimuth
 return total

def grid(ne=4,na=2,upper=24.,**controls):
 if type(ne)is not int or not 3<=ne<=6 or type(na)is not int or na not in[2,4]or not math.isfinite(upper)or not 12<=upper<=40:raise ValueError('Selected small physical energy/angular grid only')
 x,ew=leggauss(ne);en=(x+1)*upper/2;ew=ew*upper/2;mu,aw=leggauss(na);E=np.repeat(en,na);wE=np.repeat(ew,na);m=np.tile(mu,ne);wmu=np.tile(aw,ne);W=wE*E*E/(2*math.pi**2*HC**3)*wmu/2;N=len(E);K=np.zeros((N,N));start=time.monotonic()
 for i in range(N):
  for j in range(N):
   if i!=j:
    if time.monotonic()-start>100:raise TimeoutError('Angular grid work bound')
    K[i,j]=angular_edge(float(E[i]),float(E[j]),float(m[i]),float(m[j]),**controls)
 A=W[:,None]*wE[None,:]*wmu[None,:]*K
 kms=max(abs(math.log(A[i,j]/A[j,i])-(E[i]-E[j])/T)for i in range(N)for j in range(i+1,N))
 return dict(E=E.tolist(),mu=m.tolist(),wE=wE.tolist(),wmu=wmu.tolist(),W=W.tolist(),Kbar=K.tolist(),A=A.tolist(),ne=ne,na=na,upper=upper,controls=controls,kms_residual=kms,seconds=time.monotonic()-start)

def collision(g,f):
 E,W,A=_phase_arrays(g);N=len(E);f=_occupation(f,(N,),'occupation vector')
 dn=np.zeros(N);heat=entropy=activity=0.
 for i in range(N):
  for j in range(i+1,N):
   a=math.log(A[i,j])+math.log(f[i])+math.log1p(-f[j]);b=math.log(A[j,i])+math.log(f[j])+math.log1p(-f[i]);delta=a-b;J=math.copysign(math.exp(max(a,b))*(-math.expm1(-abs(delta))),delta)
   dn[i]-=J;dn[j]+=J;heat-=(E[j]-E[i])*J;entropy+=J*delta;activity+=math.exp(a)+math.exp(b)
 return dict(df=dn/W,dn=dn,heat=heat,entropy=entropy,activity=activity)

def boundary(E,left_T=3.,right_T=1.3,mu=-.7):
 E=_real_array(E,'boundary energies')
 if E.ndim!=1 or not len(E)or np.any(E<=0)or not all(_finite_real(z)for z in[left_T,right_T,mu])or min(left_T,right_T)<=0:raise ValueError('Finite positive energies/thermal reservoir temperatures and finite chemical potential required')
 left=expit((mu-E)/left_T);right=expit((mu-E)/right_T)
 return _occupation(left,E.shape,'left boundary'),_occupation(right,E.shape,'right boundary')

def rhs(g,f,left,right,*,crossing_time=.05,collisions=True):
 """Upwind streaming plus paired collision; normalized volume-average ledgers."""
 E,W,_=_phase_arrays(g);m=_real_array(g['mu'],'angular nodes');f=_real_array(f,'cell populations')
 if f.ndim!=2 or not f.shape[0]or m.shape!=E.shape or np.any(abs(m)>=1)or not _finite_real(crossing_time)or not 0<crossing_time<=1 or type(collisions)is not bool:raise ValueError('Finite matching slab arrays/scale and literal collision flag required')
 nc,N=f.shape;f=_occupation(f,(nc,len(E)),'cell populations');left=_occupation(left,(len(E),),'left boundary');right=_occupation(right,(len(E),),'right boundary')
 # Face flux μf; positive direction upwinds left, negative upwinds right.
 faces=np.empty((nc+1,N));faces[0]=m*np.where(m>0,left,f[0]);faces[-1]=m*np.where(m>0,f[-1],right)
 for z in range(1,nc):faces[z]=m*np.where(m>0,f[z-1],f[z])
 df=-(faces[1:]-faces[:-1])*nc/crossing_time;heat=entropy=0.
 if collisions:
  for z in range(nc):
   q=collision(g,f[z]);df[z]+=q['df'];heat+=q['heat']/nc;entropy+=q['entropy']/nc
 incoming=W*(np.where(m>0,m*left,-m*right))/crossing_time;outgoing=W*(np.where(m>0,m*f[-1],-m*f[0]))/crossing_time
 return df,np.array([incoming.sum(),outgoing.sum(),incoming@E,outgoing@E,heat,entropy])

def evolve(g,*,cells=3,duration=.06,crossing_time=.05,left_T=3.,right_T=1.3,rad_mu=-.7,initial_T=2.,collisions=True,rtol=2e-10,max_step=.002):
 controls=[duration,crossing_time,left_T,right_T,rad_mu,initial_T,rtol,max_step]
 if type(cells)is not int or not 1<=cells<=8 or not all(_finite_real(z)for z in controls)or not 0<duration<=.2 or not 0<crossing_time<=1 or not 0<max_step<=duration or min(left_T,right_T,initial_T,rtol)<=0 or rtol>=1 or type(collisions)is not bool:raise ValueError('Finite bounded slab controls and positive thermal temperatures/rtol required')
 E,W,_=_phase_arrays(g);N=len(E);left,right=boundary(E,left_T,right_T,rad_mu);f0=np.tile(expit((rad_mu-E)/initial_T),(cells,1));f0=_occupation(f0,(cells,N),'initial cell populations');y0=np.r_[f0.ravel(),np.zeros(6)];start=time.monotonic()
 def fun(t,y):
  if time.monotonic()-start>110:raise TimeoutError('Slab evolution work bound')
  f=y[:cells*N].reshape((cells,N));df,ledger=rhs(g,f,left,right,crossing_time=crossing_time,collisions=collisions);return np.r_[df.ravel(),ledger]
 sol=solve_ivp(fun,[0,duration],y0,method='DOP853',rtol=rtol,atol=1e-18,max_step=max_step)
 if not sol.success:raise ArithmeticError('Slab ODE failed')
 ff=sol.y[:cells*N,-1].reshape((cells,N));ledger=sol.y[cells*N:,-1];dN=float(np.mean((ff-f0)@W));dE=float(np.mean((ff-f0)@(W*E)))
 return dict(cells=cells,duration=duration,crossing_time=crossing_time,length_cm=crossing_time*C_CM_S,left_T=left_T,right_T=right_T,rad_mu=rad_mu,initial_T=initial_T,collisions=collisions,initial=f0.tolist(),final=ff.tolist(),boundary_in_number=ledger[0],boundary_out_number=ledger[1],boundary_in_energy=ledger[2],boundary_out_energy=ledger[3],thermostat_heat=ledger[4],collision_entropy=ledger[5],radiation_number_change=dN,radiation_energy_change=dE,number_ledger=dN-ledger[0]+ledger[1],energy_ledger=dE+ledger[4]-ledger[2]+ledger[3],min_occupation=float(np.min(sol.y[:cells*N])),max_occupation=float(np.max(sol.y[:cells*N])),rtol=rtol,max_step=max_step,nfev=sol.nfev,seconds=time.monotonic()-start,units='volume-average N fm^-3; E/heat MeVfm^-3; time s; length cm; one helicity axisymmetric nue')
