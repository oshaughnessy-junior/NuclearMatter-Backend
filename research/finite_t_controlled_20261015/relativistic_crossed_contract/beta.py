"""Vacuum n beta decay / inverse-current probes, not a thermal rate kernel."""
from __future__ import annotations
from pathlib import Path
import sys,math
import numpy as np
from numpy.polynomial.legendre import leggauss
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'relativistic_capture_oracle'))
from events import spinors,vertices,adj,boost,G,G5,I,MET,GF,VUD,HBAR
MN=939.56542052;MP=938.27208816;ME=.51099895069;MMU=105.6583755

def amplitude(pn,pp,l,k,mn=MN,mp=MP,ml=ME,Mweak=None,model='full',inverse=False):
 """Spin-averaged n->p e- anti amplitude, or reciprocal p e- anti->n."""
 if model not in ['full','point_VA']:raise ValueError('Known current model required')
 M=(mn+mp)/2 if Mweak is None else Mweak
 if not all(math.isfinite(v)for v in [mn,mp,ml,M])or min(mn,mp,M)<=0 or ml<0:raise ValueError('Finite masses and positive bare current mass required')
 if not isinstance(inverse,bool)or any(p.ndim!=2 or p.shape[1]!=4 or not np.all(np.isfinite(p))or np.any(p[:,0]<=0)for p in [pn,pp,l,k]):raise ValueError('Finite positive-energy four-vectors and explicit inverse bool required')
 q=pp-pn
 V=vertices(q,M)if model=='full'else np.broadcast_to(G-1.2723*(G@G5),(len(k),4,4,4));un=spinors(pn,mn);up=spinors(pp,mp);ue=spinors(l,ml);va=spinors(k,0,True);left=G@(I-G5)
 if inverse:
  # Reciprocal current, not an unchanged antineutrino-capture lepton tensor.
  J=np.einsum('nai,ab,nubc,ncj->nuij',un.conj(),G[0],adj(V),up).reshape((-1,4,4));L=np.einsum('nai,ab,ubc,ncj->nuij',va.conj(),G[0],left,ue).reshape((-1,4,4))
 else:
  J=np.einsum('nai,ab,nubc,ncj->nuij',up.conj(),G[0],V,un).reshape((-1,4,4));L=np.einsum('nai,ab,ubc,ncj->nuij',ue.conj(),G[0],left,va).reshape((-1,4,4))
 amps=np.einsum('nui,u,nuj->nij',L,MET,J);return GF**2*VUD**2/4*np.sum(abs(amps)**2,axis=(1,2))
def controls(order,mn,mp,ml,model):
 if isinstance(order,bool)or not isinstance(order,int)or order<4 or not all(math.isfinite(v)for v in [mn,mp,ml])or min(mn,mp)<=0 or ml<0 or model not in ['full','point_VA']:raise ValueError('Finite masses/known model/integer order>=4')
 if mn<=mp+ml:raise ValueError('Vacuum decay forbidden; no medium zero certificate')
def pair_events(s,c,mn,mp,ml):
 # n at rest, lepton-pair momentum along +z; full recoil and timelike pair Q.
 rt=np.sqrt(s);p=np.sqrt(((mn-mp)**2-s)*((mn+mp)**2-s))/(2*mn);EQ=(mn*mn-mp*mp+s)/(2*mn);Ep=mn-EQ;pe=(s-ml*ml)/(2*rt);Ee=(s+ml*ml)/(2*rt)
 pn=np.tile([mn,0,0,0],(len(s),1));pp=np.column_stack([Ep,np.zeros(len(s)),np.zeros(len(s)),-p]);lc=np.column_stack([Ee,pe*np.sqrt(1-c*c),np.zeros(len(s)),pe*c]);kc=np.column_stack([pe,-lc[:,1:]]);beta=np.column_stack([np.zeros(len(s)),np.zeros(len(s)),p/EQ]);return pn,pp,boost(lc,beta),boost(kc,beta),p,pe

def pair_integral(order=24,mn=MN,mp=MP,ml=ME,model='full'):
 controls(order,mn,mp,ml,model);x,w=leggauss(order);th=(x+1)*math.pi/4;dth=w*math.pi/4;sm=ml*ml;sw=(mn-mp)**2-sm;sv=sm+sw*np.sin(th)**2;ws=2*sw*np.sin(th)*np.cos(th)*dth;S,C=np.meshgrid(sv,x,indexing='ij');s=S.ravel();c=C.ravel();weight=(ws[:,None]*w[None,:]).ravel();pn,pp,l,k,p,pe=pair_events(s,c,mn,mp,ml);amp=amplitude(pn,pp,l,k,mn,mp,ml,model=model);gamma=float(np.sum(weight*p*pe*amp/(128*math.pi**3*mn*mn*np.sqrt(s)))/HBAR)
 return dict(rate=gamma,lifetime=1/gamma,energy_neutrino=float(np.sum(weight*p*pe*amp*k[:,0]/(128*math.pi**3*mn*mn*np.sqrt(s)))/HBAR),energy_electron=float(np.sum(weight*p*pe*amp*l[:,0]/(128*math.pi**3*mn*mn*np.sqrt(s)))/HBAR),energy_proton=float(np.sum(weight*p*pe*amp*pp[:,0]/(128*math.pi**3*mn*mn*np.sqrt(s)))/HBAR),shell_residual=float(np.max(abs(pn-pp-l-k))/mn))

def electron_events(E,c,mn,mp,ml):
 pe=np.sqrt((E-ml)*(E+ml));Emax=((mn-mp)*(mn+mp)+ml*ml)/(2*mn);nu=mn*(Emax-E)/(mn-E+pe*c);Ep=mn-E-nu;pn=np.tile([mn,0,0,0],(len(E),1));l=np.column_stack([E,np.zeros(len(E)),np.zeros(len(E)),pe]);k=np.column_stack([nu,nu*np.sqrt(1-c*c),np.zeros(len(E)),nu*c]);pp=pn-l-k
 return pn,pp,l,k,pe,nu,(mn-E+pe*c)
def electron_integral(order=24,mn=MN,mp=MP,ml=ME,model='full'):
 controls(order,mn,mp,ml,model);x,w=leggauss(order);Emax=((mn-mp)*(mn+mp)+ml*ml)/(2*mn);end=math.sqrt(Emax-ml);z=(x+1)*end/2;Ee=ml+z*z;we=w*end*z;E,C=np.meshgrid(Ee,x,indexing='ij');e=E.ravel();c=C.ravel();weight=(we[:,None]*w[None,:]).ravel();pn,pp,l,k,pe,nu,den=electron_events(e,c,mn,mp,ml);amp=amplitude(pn,pp,l,k,mn,mp,ml,model=model);base=weight*amp*pe*nu/(64*math.pi**3*mn*den)/HBAR;gamma=float(np.sum(base))
 return dict(rate=gamma,lifetime=1/gamma,energy_neutrino=float(np.sum(base*nu)),energy_electron=float(np.sum(base*e)),energy_proton=float(np.sum(base*pp[:,0])),shell_residual=float(np.max(abs(pp[:,0]**2-np.sum(pp[:,1:]**2,axis=1)-mp*mp))/(mp*mp)))
def spectrum(E,order=32,mn=MN,mp=MP,ml=ME,model='full'):
 controls(order,mn,mp,ml,model);Emax=((mn-mp)*(mn+mp)+ml*ml)/(2*mn)
 if not math.isfinite(E)or not ml<E<Emax:raise ValueError('Interior physical total electron energy required')
 c,w=leggauss(order);e=np.full(order,E);pn,pp,l,k,pe,nu,den=electron_events(e,c,mn,mp,ml);amp=amplitude(pn,pp,l,k,mn,mp,ml,model=model)
 return float(np.sum(w*amp*pe*nu/(64*math.pi**3*mn*den))/HBAR)
