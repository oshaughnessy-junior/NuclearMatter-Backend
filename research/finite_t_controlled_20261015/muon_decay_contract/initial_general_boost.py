"""Tree-level unpolarized mu- -> e- anti-nu_e nu_mu, no Vud/QED."""
from __future__ import annotations
from pathlib import Path
import sys,math
import numpy as np
from numpy.polynomial.legendre import leggauss
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'relativistic_capture_oracle'))
from events import spinors,boost,G,G5,I,MET,GF,HBAR
MMU=105.6583755;ME=.51099895069

def dot(a,b):return np.sum(a*b*MET,axis=1)
def matrix_squared(pm,e,ae,nm,m=MMU,me=ME,method='spinor',inverse=False):
 if not all(math.isfinite(x)for x in[m,me])or m<=0 or me<0 or not isinstance(inverse,bool):raise ValueError('Positive parent/nonnegative daughter masses and bool inverse')
 if any(p.ndim!=2 or p.shape[1]!=4 or not np.all(np.isfinite(p))or np.any(p[:,0]<=0)for p in[pm,e,ae,nm]):raise ValueError('Finite positive-energy four-vectors required')
 if method=='invariant':return 64*GF**2*dot(pm,ae)*dot(e,nm)
 if method!='spinor':raise ValueError('Known matrix method required')
 um=spinors(pm,m);ue=spinors(e,me);ua=spinors(ae,0,True);un=spinors(nm,0);left=G@(I-G5)
 if inverse:
  J=np.einsum('nai,ab,ubc,ncj->nuij',um.conj(),G[0],left,un).reshape((-1,4,4));L=np.einsum('nai,ab,ubc,ncj->nuij',ua.conj(),G[0],left,ue).reshape((-1,4,4))
 else:
  J=np.einsum('nai,ab,ubc,ncj->nuij',un.conj(),G[0],left,um).reshape((-1,4,4));L=np.einsum('nai,ab,ubc,ncj->nuij',ue.conj(),G[0],left,ua).reshape((-1,4,4))
 amps=np.einsum('nui,u,nuj->nij',J,MET,L);return GF**2/4*np.sum(abs(amps)**2,axis=(1,2))
def controls(order,m,me):
 if isinstance(order,bool)or not isinstance(order,int)or order<4 or not all(math.isfinite(x)for x in[m,me])or me<0 or m<=me:raise ValueError('Finite m>me>=0 and integer order>=4 required')
def analytic_rate(m=MMU,me=ME):
 controls(4,m,me);r=(me/m)**2
 if r==0:f=1.
 elif r>.5:
  d=1-r;f=math.fsum(24*d**n/(n*(n-1)*(n-2))for n in range(5,180))
 else:f=1-8*r+8*r**3-r**4-12*r*r*math.log(r)
 if f<=0:raise ArithmeticError('Cancellation-prone near-equal-mass analytic expression unresolved')
 return GF**2*m**5/(192*math.pi**3*HBAR)*f

def pair_events(s,c,m,me):
 p=np.sqrt(((m-me)**2-s)*((m+me)**2-s))/(2*m);Ee=(m*m+me*me-s)/(2*m);EQ=m-Ee;rt=np.sqrt(s);kn=rt/2;pm=np.tile([m,0,0,0],(len(s),1));e=np.column_stack([Ee,np.zeros(len(s)),np.zeros(len(s)),-p]);ac=np.column_stack([kn,kn*np.sqrt(1-c*c),np.zeros(len(s)),kn*c]);nc=np.column_stack([kn,-ac[:,1:]]);beta=np.column_stack([np.zeros(len(s)),np.zeros(len(s)),p/EQ]);return pm,e,boost(ac,beta),boost(nc,beta),p

def summarize(base,pm,e,ae,nm):
 rate=float(np.sum(base));return dict(rate=rate,lifetime=1/rate,energy_e=float(np.sum(base*e[:,0])),energy_anti_e=float(np.sum(base*ae[:,0])),energy_nu_mu=float(np.sum(base*nm[:,0])),fourmomentum_residual=float(np.max(abs(pm-e-ae-nm))/pm[0,0]))
def pair_integral(order=24,m=MMU,me=ME,method='invariant'):
 controls(order,m,me);x,w=leggauss(order);th=(x+1)*math.pi/4;ds=(m-me)**2;sv=ds*np.sin(th)**2;ws=ds*2*np.sin(th)*np.cos(th)*w*math.pi/4;S,C=np.meshgrid(sv,x,indexing='ij');s=S.ravel();c=C.ravel();weights=(ws[:,None]*w[None,:]).ravel();pm,e,ae,nm,p=pair_events(s,c,m,me);amp=matrix_squared(pm,e,ae,nm,m,me,method);base=weights*p*amp/(256*math.pi**3*m*m*HBAR);return summarize(base,pm,e,ae,nm)

def dalitz_events(Ee,Ea,m,me):
 pe=np.sqrt((Ee-me)*(Ee+me));Emax=(m*m+me*me)/(2*m);c=(m*(Emax-Ee)/Ea-(m-Ee))/pe
 if np.any(abs(c)>1+1e-10):raise ArithmeticError('Unphysical Dalitz angle')
 pm=np.tile([m,0,0,0],(len(Ee),1));e=np.column_stack([Ee,np.zeros(len(Ee)),np.zeros(len(Ee)),pe]);ae=np.column_stack([Ea,Ea*np.sqrt(np.maximum(0,1-c*c)),np.zeros(len(Ee)),Ea*c]);nm=pm-e-ae;return pm,e,ae,nm

def dalitz_integral(order=24,m=MMU,me=ME,method='spinor'):
 controls(order,m,me);x,w=leggauss(order);Emax=(m*m+me*me)/(2*m);z=(x+1)/2*math.sqrt(Emax-me);Ee=me+z*z;we=w*math.sqrt(Emax-me)*z;E,C=np.meshgrid(Ee,x,indexing='ij');ee=E.ravel();pe=np.sqrt((ee-me)*(ee+me));lo=(m-ee-pe)/2;hi=(m-ee+pe)/2;Ea=(lo+hi)/2+(hi-lo)/2*C.ravel();weights=(we[:,None]*w[None,:]).ravel()*(hi-lo)/2;pm,e,ae,nm=dalitz_events(ee,Ea,m,me);amp=matrix_squared(pm,e,ae,nm,m,me,method);base=weights*amp/(64*math.pi**3*m*HBAR);return summarize(base,pm,e,ae,nm)

def spectrum(E,order=32,m=MMU,me=ME,method='spinor'):
 controls(order,m,me);Emax=(m*m+me*me)/(2*m)
 if not math.isfinite(E)or not me<E<Emax:raise ValueError('Interior total electron energy required')
 x,w=leggauss(order);ee=np.full(order,E);pe=math.sqrt((E-me)*(E+me));lo=(m-E-pe)/2;hi=(m-E+pe)/2;Ea=(lo+hi)/2+(hi-lo)/2*x;pm,e,ae,nm=dalitz_events(ee,Ea,m,me);amp=matrix_squared(pm,e,ae,nm,m,me,method);return float(np.sum(w*(hi-lo)/2*amp/(64*math.pi**3*m*HBAR)))
