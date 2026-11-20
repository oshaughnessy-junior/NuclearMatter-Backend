"""Independent moving-target CM event/spinor total capture oracle."""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
from scipy.special import log_expit,logsumexp
from numpy.polynomial.legendre import leggauss
GF=1.1663785e-11;VUD=.974;HBAR=6.582119569e-22;HC=197.3269804
S=np.array([[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]],complex);Z=np.zeros((2,2),complex);G=np.array([np.diag([1.,1.,-1.,-1.])]+[np.block([[Z,s],[-s,Z]])for s in S]);G5=1j*G[0]@G[1]@G[2]@G[3];I=np.eye(4);MET=np.array([1.,-1.,-1.,-1.])
@dataclass(frozen=True)
class Bath:
 T:float;ma:float;mb:float;mua:float;mub:float;mul:float;Enu:float=150.;ml:float=105.6583755;anti:bool=False;Mweak:float=938.91875434
 def __post_init__(self):
  if not all(math.isfinite(v)for k,v in vars(self).items()if k!='anti')or min(self.T,self.ma,self.mb,self.Enu,self.Mweak)<=0 or self.ml<0 or not isinstance(self.anti,bool):raise ValueError('Finite physical bath required')

def spinors(p,m,anti=False):
 root=np.sqrt(p[:,0]+m);sig=np.einsum('ni,ijk->njk',p[:,1:],S);big=root[:,None,None]*np.eye(2);small=sig/root[:,None,None]
 return np.concatenate([small,big]if anti else [big,small],axis=1)
def adj(v):return np.einsum('ab,nubc,cd->nuad',G[0],v.conj().swapaxes(-1,-2),G[0])
def vertices(q,M):
 t=q[:,0]**2-np.sum(q[:,1:]**2,axis=1)
 if np.any(t>=min(139.57**2,840.**2,1000.**2,4*M*M)):raise ValueError('Sub-pole current required')
 den=(1-t/(4*M*M))*(1-t/840.**2)**2;v=(1-4.706*t/(4*M*M))/den;a=1.2723/(1-t/1000.**2)**2;f=3.706/den;gp=2*M*M*a/(139.57**2-t)
 comm=np.array([[G[u]@G[j]-G[j]@G[u]for j in range(4)]for u in range(4)])
 return v[:,None,None,None]*G[None,:,:,:]-a[:,None,None,None]*(G@G5)[None,:,:,:]-f[:,None,None,None]/(4*M)*np.einsum('nj,j,ujab->nuab',q,MET,comm)-gp[:,None,None,None]/M*q[:,:,None,None]*G5[None,None,:,:]
def amplitude(b,k,pa,l,pb):
 ua=spinors(pa,b.ma);ub=spinors(pb,b.mb);uk=spinors(k,0,b.anti);ul=spinors(l,b.ml,b.anti);q=k-l;V=vertices(q,b.Mweak)if not b.anti else adj(vertices(-q,b.Mweak));left=G@(I-G5)
 J=np.einsum('nai,ab,nubc,ncj->nuij',ub.conj(),G[0],V,ua).reshape((-1,4,4))
 if b.anti:L=np.einsum('nai,ab,ubc,ncj->nuij',uk.conj(),G[0],left,ul).reshape((-1,4,4))
 else:L=np.einsum('nai,ab,ubc,ncj->nuij',ul.conj(),G[0],left,uk).reshape((-1,4,4))
 amps=np.einsum('nui,u,nuj->nij',L,MET,J);return GF**2*VUD**2/4*np.sum(abs(amps)**2,axis=(1,2))
def boost(v,beta):
 b2=np.sum(beta*beta,axis=1);gamma=1/np.sqrt(1-b2);dot=np.sum(beta*v[:,1:],axis=1);E=gamma*(v[:,0]+dot);factor=gamma*v[:,0]+gamma**2/(gamma+1)*dot
 return np.column_stack([E,v[:,1:]+factor[:,None]*beta])
def represented(logx):
 if not math.isfinite(logx):raise ArithmeticError('Unresolved finite logarithm')
 try:v=math.exp(logx)
 except OverflowError:return dict(log=logx,value=None,representable=False,reason='overflow')
 return dict(log=logx,value=v if v>0 else None,representable=v>0,reason='represented'if v>0 else 'underflow')

def integrate(b,radial=10,incident=10,outgoing=16,azimuth=8,tail=40.,checkpoint=None):
 for x in [radial,incident,outgoing,azimuth]:
  if isinstance(x,bool)or not isinstance(x,int)or x<4:raise ValueError('Integer quadrature controls>=4')
 if not math.isfinite(tail)or not 20<=tail<=100:raise ValueError('Finite tail20..100')
 xr,wr=leggauss(radial);ci,wi=leggauss(incident);co,wo=leggauss(outgoing);phi=2*math.pi*np.arange(azimuth)/azimuth
 CI,CO,PHI=np.meshgrid(ci,co,phi,indexing='ij');ci=CI.ravel();co=CO.ravel();phi=PHI.ravel();aw=(wi[:,None,None]*wo[None,:,None]*np.ones((1,1,azimuth))*2*math.pi/azimuth).ravel()
 center=max(0,b.mua-b.ma);end=center+tail*b.T;cuts=sorted(set([0.,end]+[v for v in [b.T,3*b.T,8*b.T,16*b.T,center-8*b.T,center,center+8*b.T]if 0<v<end]));logs=[[],[]];density=0.;maxres=0.;N=len(ci);empty=0
 for ix,(lo,hi)in enumerate(zip(cuts[:-1],cuts[1:])):
  xlo,xhi=math.sqrt(lo),math.sqrt(hi)
  xx=(xlo+xhi)/2+(xhi-xlo)*xr/2
  for kinetic,rw in zip(xx*xx,wr*(xhi-xlo)*xx):
   Ea=b.ma+kinetic;p=math.sqrt(kinetic*(kinetic+2*b.ma));density+=p*Ea*rw/(math.pi**2*HC**3)*math.exp(float(log_expit((b.mua-Ea)/b.T)))
   k=np.tile([b.Enu,0,0,b.Enu],(N,1));pa=np.column_stack([np.full(N,Ea),p*np.sqrt(1-ci*ci),np.zeros(N),p*ci]);P=k+pa;s=P[:,0]**2-np.sum(P[:,1:]**2,axis=1);valid=s>(b.mb+b.ml)**2
   if not np.any(valid):empty+=1;continue
   kv=k[valid];pav=pa[valid];sv=s[valid];beta=P[valid,1:]/P[valid,0,None];cmk=boost(kv,-beta);axis=cmk[:,1:]/np.linalg.norm(cmk[:,1:],axis=1)[:,None];e2=np.tile([0,1,0],(len(sv),1));e1=np.cross(e2,axis)
   rt=np.sqrt(sv);pi=(sv-b.ma*b.ma)/(2*rt);pf=np.sqrt((sv-(b.mb+b.ml)**2)*(sv-(b.mb-b.ml)**2))/(2*rt);El=(sv+b.ml*b.ml-b.mb*b.mb)/(2*rt);outcos=co[valid];outphi=phi[valid];direc=outcos[:,None]*axis+np.sqrt(1-outcos*outcos)[:,None]*(np.cos(outphi)[:,None]*e1+np.sin(outphi)[:,None]*e2)
   lcm=np.column_stack([El,pf[:,None]*direc]);bcm=np.column_stack([rt-El,-pf[:,None]*direc]);l=boost(lcm,beta);pb=boost(bcm,beta);amp=amplitude(b,kv,pav,l,pb);sigma=amp/(64*math.pi**2*sv)*pf/pi;flux=(Ea-p*ci[valid])/Ea;weight=p*Ea*rw/(2*math.pi**2)*aw[valid]*flux*sigma/HBAR
   if np.any(weight<=0)or np.any(~np.isfinite(weight)):raise ArithmeticError('Positive event weight unresolved')
   fa=float(log_expit((b.mua-Ea)/b.T));nfa=float(log_expit((Ea-b.mua)/b.T));fb=log_expit((b.mub-pb[:,0])/b.T);nfb=log_expit((pb[:,0]-b.mub)/b.T);fl=log_expit((b.mul-l[:,0])/b.T);nfl=log_expit((l[:,0]-b.mul)/b.T)
   logs[0].append(float(logsumexp(np.log(weight)+fa+nfb+nfl)));logs[1].append(float(logsumexp(np.log(weight)+fb+fl+nfa)));res=kv+pav-l-pb;maxres=max(maxres,float(np.max(abs(res)))/max(b.Enu,b.ma))
  if checkpoint:checkpoint(dict(panel=ix,panels=len(cuts)-1,density=density,chi=represented(float(logsumexp(logs[0]))),j=represented(float(logsumexp(logs[1])))))
 if not logs[0]:raise ArithmeticError('Empty represented event support; no zero certificate')
 return dict(chi=represented(float(logsumexp(logs[0]))),j=represented(float(logsumexp(logs[1]))),density=density,max_fourmomentum_residual=maxres,empty_radial_nodes=empty,controls=dict(radial=radial,incident=incident,outgoing=outgoing,azimuth=azimuth,tail=tail))
WARM=Bath(2,939.56542052,938.27208816,933.6807508996585,930.6761152463529,97.74668143468162)
DILUTE=Bath(.2,939.56542052,938.27208816,934.5991075502667,933.136728749228,0.)
