"""Free e- e+ <-> nu_a anti_nu_a, exact me Fermi-limit reference.

Unpolarized initial electron/positron spin average4; no Vud, one active ν helicity.
"""
from pathlib import Path
from dataclasses import dataclass
import sys,math,time
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import log_expit,logsumexp
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'relativistic_capture_oracle'));import events as tools
GF=1.1663785e-11;HBAR=6.582119569e-22;HC=197.3269804;ME=.51099895069
G=tools.G;G5=tools.G5;I=tools.I;MET=tools.MET
@dataclass(frozen=True)
class Couplings:
 flavor:str
 sin2w:float=.23129
 def __post_init__(self):
  if self.flavor not in ('e','mu','tau')or not math.isfinite(self.sin2w)or not 0<self.sin2w<1:raise ValueError('Named flavor and fixed sin²thetaW in(0,1) required')
 @property
 def cv(self):return (.5 if self.flavor=='e'else-.5)+2*self.sin2w
 @property
 def ca(self):return .5 if self.flavor=='e'else-.5
@dataclass(frozen=True)
class Radiation:
 temperature:float
 mu_nu:float
 mu_antinu:float
 def __post_init__(self):
  if not all(math.isfinite(x)for x in vars(self).values())or self.temperature<=0:raise ValueError('Finite FD partner inputs and positive temperature required')
def dot(a,b):return np.sum(a*b*MET,axis=-1)
def slash(p):return np.einsum('nu,u,uab->nab',p,MET,G)
def adj(m):return G[0]@m.conj().T@G[0]
def represented(log):
 if log is None:return dict(value=0.,log=None,representable=True,reason='empty_occupation')
 if not math.isfinite(log):raise ArithmeticError('Finite positive moment log required')
 with np.errstate(over='ignore',under='ignore'):v=float(np.exp(log))
 return dict(log=log,value=v if math.isfinite(v)and v>0 else None,representable=bool(math.isfinite(v)and v>0),reason=None if math.isfinite(v)and v>0 else ('overflow'if v else'underflow'))
def amplitude(pm,pp,k,a,coupling,*,mass=ME,method='scalar'):
 if method=='scalar':
  v,g=coupling.cv,coupling.ca
  return 8*GF*GF*((g-v)**2*dot(pm,k)*dot(pp,a)+(g+v)**2*dot(pp,k)*dot(pm,a)+mass*mass*(v*v-g*g)*dot(k,a))
 if method in ('spinor','inverse_spinor'):
  um=tools.spinors(pm,mass);vp=tools.spinors(pp,mass,True);uk=tools.spinors(k,0);va=tools.spinors(a,0,True)
  left=G@(I-G5);el=G@(coupling.cv*I-coupling.ca*G5)
  if method=='inverse_spinor':
   left=np.array([adj(x)for x in left]);el=np.array([adj(x)for x in el]);u1,u2=va,uk;e1,e2=um,vp
  else:u1,u2=uk,va;e1,e2=vp,um
  J=np.einsum('nai,ab,ubc,ncj->nuij',u1.conj(),G[0],left,u2).reshape((-1,4,4))
  E=np.einsum('nai,ab,ubc,ncj->nuij',e1.conj(),G[0],el,e2).reshape((-1,4,4))
  A=np.einsum('nui,u,nuj->nij',J,MET,E)
  # Common phase-space kernel averages charged spins4 in either direction;
  # inverse physical active-neutrino initial average1 has amplitude4times this.
  return GF*GF/8*np.sum(abs(A)**2,axis=(1,2))
 if method!='trace':raise ValueError('scalar,trace or explicit reciprocal spinors required')
 left=G@(I-G5);el=G@(coupling.cv*I-coupling.ca*G5)
 sm=slash(pm)+mass*I;sp=slash(pp)-mass*I;sk=slash(k);sa=slash(a)
 L=np.array([[np.einsum('nij,ji->n',sk@left[u]@sa,adj(left[v]))for v in range(4)]for u in range(4)])
 # Sum both initial charged spins, then average4 explicitly; do not average ν.
 H=np.array([[np.einsum('nij,ji->n',sp@el[u]@sm,adj(el[v]))for v in range(4)]for u in range(4)])
 z=GF*GF/8*np.einsum('u,v,uvn,uvn->n',MET,MET,L,H)
 if np.max(abs(z.imag))>1e-10*max(np.max(abs(z.real)),1e-300):raise ArithmeticError('Nonreal annihilation amplitude')
 return z.real
def cm(s,c,mass=ME):
 if s<=4*mass*mass or not -1<=c<=1:raise ValueError('Above threshold and physical CM cosine required')
 E=math.sqrt(s)/2;p=math.sqrt((E-mass)*(E+mass));q=E
 return tuple(np.array([z],float)for z in [[E,0,0,p],[E,0,0,-p],[q,q*math.sqrt(1-c*c),0,q*c],[q,-q*math.sqrt(1-c*c),0,-q*c]])
def sigma(s,coupling,mass=ME):
 beta=math.sqrt(1-4*mass*mass/s);r=mass*mass/s
 return GF*GF*s/(12*math.pi*beta)*(coupling.cv**2*(1+2*r)+coupling.ca**2*(1-4*r))
def sigma_v(pm,pp,coupling,mass=ME):
 s=2*mass*mass+2*dot(pm,pp)
 return GF*GF*s/(24*math.pi*pm[:,0]*pp[:,0])*(s*(coupling.cv**2+coupling.ca**2)+mass*mass*(2*coupling.cv**2-4*coupling.ca**2))

def thermal(T,mu,coupling,*,radial=20,incident=8,outgoing=8,azimuth=8,tail=40.,mass=ME,budget_seconds=110.,radiation=None):
 """Empty partners or explicit BOTH common-T FD neutrino partner occupations."""
 if not all(math.isfinite(x)for x in [T,mu,tail,mass,budget_seconds])or T<=0 or mass<0 or not 20<=tail<=80 or not 0<budget_seconds<=120:raise ValueError('Finite physical inputs and boundedtail/budget required')
 if any(isinstance(n,bool)or not isinstance(n,int)or n<4 for n in [radial,incident,outgoing,azimuth]):raise ValueError('Integer orders>=4 required')
 if radiation is not None and(not isinstance(radiation,Radiation)or radiation.temperature!=T):raise ValueError('Explicit common-material-T FD radiation or None required')
 x,w=leggauss(radial);ci,wi=leggauss(incident);co,wo=leggauss(outgoing);phi=2*math.pi*np.arange(azimuth)/azimuth
 CI,CO,PH=np.meshgrid(ci,co,phi,indexing='ij');CI=CI.ravel();CO=CO.ravel();PH=PH.ravel()
 aw=(wi[:,None,None]*wo[None,:,None]*np.ones((1,1,azimuth))*2*math.pi/azimuth).ravel();N=len(CI)
 center=max(0.,abs(mu)-mass)
 if radiation is not None:center=max(center,radiation.mu_nu+radiation.mu_antinu-2*mass)
 upper=center+tail*T
 # Independent charged-particle radial measures spin2 p²dp/pi².
 cuts=[0.,T,3*T,8*T,16*T,upper];cuts=sorted(set(z for z in cuts if 0<=z<=upper));nodes=[]
 for lo,hi in zip(cuts[:-1],cuts[1:]):
  a,b=math.sqrt(lo),math.sqrt(hi)
  for xx,ww in zip(x,w):
   y=a+(xx+1)*(b-a)/2;kin=y*y;E=mass+kin;p=math.sqrt(kin*(kin+2*mass));dK=ww*(b-a)*y
   nodes.append((E,p,p*E*dK))
 logs={name:[]for name in ['number','nu_energy','antinu_energy','electron_energy','positron_energy']};backlogs={name:[]for name in logs};nets=np.zeros(5);entropy=0.;affinity_error=0.;density=np.zeros(2);anchor_number=anchor_energy=0.;max_res=0.;start=time.monotonic()
 for E,p,rw in nodes:
  fm=float(log_expit((mu-E)/T));density[0]+=rw*math.exp(fm)/(math.pi**2*HC**3)
  density[1]+=rw*math.exp(float(log_expit((-mu-E)/T)))/(math.pi**2*HC**3)
  for Ep,pp,rwp in nodes:
   if time.monotonic()-start>budget_seconds:raise TimeoutError('Pair annihilation selected total exceeded work budget')
   fp=float(log_expit((-mu-Ep)/T))
   pm=np.tile([E,0,0,p],(N,1));pv=np.column_stack([np.full(N,Ep),pp*np.sqrt(1-CI*CI),np.zeros(N),pp*CI])
   P=pm+pv;s=P[:,0]**2-np.sum(P[:,1:]**2,axis=1);rt=np.sqrt(s);beta=P[:,1:]/P[:,0,None]
   incoming=tools.boost(pm,-beta);axis=incoming[:,1:]/np.linalg.norm(incoming[:,1:],axis=1)[:,None]
   # Incoming electron CM axis; xz-plane event, fixed transverse y axis.
   yaxis=np.tile([0,1,0],(N,1));xaxis=np.cross(yaxis,axis)
   direction=CO[:,None]*axis+np.sqrt(1-CO*CO)[:,None]*(np.cos(PH)[:,None]*xaxis+np.sin(PH)[:,None]*yaxis)
   kc=np.column_stack([rt/2,rt[:,None]/2*direction]);ac=np.column_stack([rt/2,-rt[:,None]/2*direction]);k=tools.boost(kc,beta);a=tools.boost(ac,beta)
   amp=amplitude(pm,pv,k,a,coupling,mass=mass)
   # Møller velocity*dσ/dΩ: pi cancels exactly, pf=√s/2.
   event=amp/(128*math.pi**2*E*Ep)
   measure=rw*rwp/(2*math.pi**4*HBAR*HC**3)*aw
   base=measure*event
   if np.any(base<=0):raise ArithmeticError('Positive exact pair event weight required')
   lf=np.log(base)+fm+fp
   if radiation is None:net=np.exp(lf);lb=None
   else:
    fnu=log_expit((radiation.mu_nu-k[:,0])/T);nfnu=log_expit((k[:,0]-radiation.mu_nu)/T)
    fa=log_expit((radiation.mu_antinu-a[:,0])/T);nfa=log_expit((a[:,0]-radiation.mu_antinu)/T)
    F=fm+fp+nfnu+nfa
    B=float(log_expit((E-mu)/T)+log_expit((Ep+mu)/T))+fnu+fa
    delta=F-B;target=-(radiation.mu_nu+radiation.mu_antinu)/T
    affinity_error=max(affinity_error,float(np.max(abs(delta-target))))
    lf=np.log(base)+F;lb=np.log(base)+B
    net=np.sign(delta)*np.exp(np.log(base)+np.maximum(F,B))*(-np.expm1(-abs(delta)))
    entropy+=float(np.sum(net*delta))
   for i,(name,v)in enumerate([('number',np.ones(N)),('nu_energy',k[:,0]),('antinu_energy',a[:,0]),('electron_energy',pm[:,0]),('positron_energy',pv[:,0])]):
    logs[name].append(float(logsumexp(lf+np.log(v))));nets[i]+=float(np.sum(net*v))
    if lb is not None:backlogs[name].append(float(logsumexp(lb+np.log(v))))
   max_res=max(max_res,float(np.max(abs(pm+pv-k-a))/max(E,Ep)))
   # Independent final-angle integrated invariant cross-section anchor.
   pm0=pm.reshape((incident,outgoing,azimuth,4))[:,0,0];pv0=pv.reshape((incident,outgoing,azimuth,4))[:,0,0]
   av=sigma_v(pm0,pv0,coupling,mass)
   val=rw*rwp/(2*math.pi**4*HBAR*HC**3)*float(np.sum(wi*av))*math.exp(fm+fp)
   anchor_number+=val;anchor_energy+=val*(E+Ep)
 totals={name:represented(float(logsumexp(v)))for name,v in logs.items()}
 backward={name:represented(float(logsumexp(v))if v else None)for name,v in backlogs.items()}
 return dict(T=T,mu_e=mu,mass_e=mass,flavor=coupling.flavor,sin2w=coupling.sin2w,cv=coupling.cv,ca=coupling.ca,totals=totals,forward=totals,backward=backward,net=dict(zip(logs,map(float,nets))),entropy=entropy if radiation else None,max_affinity_error=affinity_error,electron_density=float(density[0]),positron_density=float(density[1]),anchor_number=anchor_number,anchor_total_energy=anchor_energy,max_fourmomentum_residual=max_res,runtime=time.monotonic()-start,controls=dict(radial=radial,incident=incident,outgoing=outgoing,azimuth=azimuth,tail=tail),radiation=vars(radiation)if radiation else 'both neutrino occupations EXACTLY empty; inverse absent',units='event number fm^-3 s^-1; energies MeV fm^-3 s^-1')
