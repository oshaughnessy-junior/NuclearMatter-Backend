"""Free massive nu/anti+e± elastic weak scattering; selected FD bath reference."""
from pathlib import Path
import sys,math,time
import numpy as np
from scipy.special import expit,log_expit
from numpy.polynomial.legendre import leggauss
R=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True
sys.path.insert(0,str(R/'leptonic_pair_annihilation'));import pair as P
GF=P.GF;ME=P.ME;HC=P.HC;HBAR=P.HBAR;G=P.G;I=P.I;G5=P.G5;MET=P.MET

def dot(a,b):return np.sum(a*b*MET,axis=1)
def amplitude(k,p,kp,pp,coupling,*,positron=False,anti=False,method='scalar'):
 if not isinstance(positron,bool)or not isinstance(anti,bool):raise ValueError('Literal lepton/neutrino charge flags required')
 v,a=coupling.cv,coupling.ca
 if method=='scalar':
  L,R=(v+a)**2,(v-a)**2
  if positron!=anti:L,R=R,L
  return 16*GF**2*(L*dot(k,p)*dot(kp,pp)+R*dot(k,pp)*dot(kp,p)-ME**2*(v*v-a*a)*dot(k,kp))
 if method!='spinor':raise ValueError('scalar or literal spinor method')
 uk=P.tools.spinors(k,0,anti);uo=P.tools.spinors(kp,0,anti);up=P.tools.spinors(p,ME,positron);uf=P.tools.spinors(pp,ME,positron)
 left=G@(I-G5);el=G@(v*I-a*G5)
 a1,a2=(uk,uo)if anti else(uo,uk);b1,b2=(up,uf)if positron else(uf,up)
 J=np.einsum('nai,ab,ubc,ncj->nuij',a1.conj(),G[0],left,a2).reshape((-1,4,4));C=np.einsum('nai,ab,ubc,ncj->nuij',b1.conj(),G[0],el,b2).reshape((-1,4,4));M=np.einsum('nui,u,nuj->nij',J,MET,C)
 return GF**2/4*np.sum(abs(M)**2,axis=(1,2)) # GF²/2 and targetspinaverage2; NO incidentνaverage/CKM

def vacuum_dsigma_drecoil(E,recoil,coupling,positron=False,anti=False):
 L,R=(coupling.cv+coupling.ca)**2,(coupling.cv-coupling.ca)**2
 if positron!=anti:L,R=R,L
 return GF**2*ME/(2*math.pi)*(L+R*(1-recoil/E)**2-(coupling.cv**2-coupling.ca**2)*ME*recoil/E**2)

def thermal(T,mu_e,Ein,coupling,*,positron=False,anti=False,radial=24,incident=12,outgoing=12,phi=12,tail=50.,radiation_mu=None,budget=110.):
 if not all(math.isfinite(z)for z in [T,mu_e,Ein,tail,budget])or T<=0 or Ein<=0 or not 20<=tail<=100 or not 0<budget<=120:raise ValueError('Finite positive bath/incident and explicit tail20..100,budget<=120')
 if not .2<=T<=4 or abs(mu_e/T)>3 or Ein>50:raise ValueError('Selected pair-rich T.2..4,|mu/T|<=3,Ein<=50 only')
 if not isinstance(positron,bool)or not isinstance(anti,bool)or not isinstance(coupling,P.Couplings):raise ValueError('Named coupling and literal charge flags required')
 if radiation_mu is not None and(not math.isfinite(radiation_mu)or abs(radiation_mu/T)>10):raise ValueError('Finite common-T outgoingν chemicalpotential orNone')
 for n in [radial,incident,outgoing,phi]:
  if isinstance(n,bool)or not isinstance(n,int)or n<4:raise ValueError('Integer orders>=4')
 x,w=leggauss(radial);ci,wi=leggauss(incident);co,wo=leggauss(outgoing);ph=2*math.pi*np.arange(phi)/phi;CI,CO,PH=np.meshgrid(ci,co,ph,indexing='ij');ci=CI.ravel();co=CO.ravel();ph=PH.ravel();aw=(wi[:,None,None]*wo[None,:,None]*np.ones((1,1,phi))*2*math.pi/phi).ravel();N=len(ci);mu=-mu_e if positron else mu_e
 end=max(0.,mu-ME)+tail*T;cuts=sorted(set([0.,T,3*T,8*T,16*T,end]));rate=transport=dE=outE=0.;density=0.;maxres=0.;start=time.monotonic();hist=np.zeros(8);edges=np.array([0.,2.,4.,6.,8.,10.,15.,25.,1000.])
 for lo,hi in zip(cuts[:-1],cuts[1:]):
  aa,bb=math.sqrt(lo),math.sqrt(hi)
  for xx,ww in zip(x,w):
   if time.monotonic()-start>budget:raise TimeoutError('Bounded lepton scattering total')
   y=aa+(xx+1)*(bb-aa)/2;kin=y*y;e=ME+kin;p=math.sqrt(kin*(kin+2*ME));rw=ww*(bb-aa)*y;f=float(expit((mu-e)/T));density+=p*e*rw*f/(math.pi**2*HC**3)
   k=np.tile([Ein,0,0,Ein],(N,1));pv=np.column_stack([np.full(N,e),p*np.sqrt(1-ci*ci),np.zeros(N),p*ci]);tot=k+pv;s=tot[:,0]**2-np.sum(tot[:,1:]**2,axis=1);rt=np.sqrt(s);beta=tot[:,1:]/tot[:,0,None];kc=P.tools.boost(k,-beta);axis=kc[:,1:]/np.linalg.norm(kc[:,1:],axis=1)[:,None];yy=np.tile([0,1,0],(N,1));ax=np.cross(yy,axis);pc=(s-ME**2)/(2*rt);direc=co[:,None]*axis+np.sqrt(1-co*co)[:,None]*(np.cos(ph)[:,None]*ax+np.sin(ph)[:,None]*yy)
   kp=P.tools.boost(np.column_stack([pc,pc[:,None]*direc]),beta);pp=P.tools.boost(np.column_stack([(s+ME**2)/(2*rt),-pc[:,None]*direc]),beta);amp=amplitude(k,pv,kp,pp,coupling,positron=positron,anti=anti);flux=(e-p*ci)/e
   weight=p*e*rw/(2*math.pi**2)*aw*flux*amp/(64*math.pi**2*s)/HBAR*f*expit((pp[:,0]-mu)/T)
   if radiation_mu is not None:weight*=expit((kp[:,0]-radiation_mu)/T)
   if np.any(weight<0)or np.any(~np.isfinite(weight)):raise ArithmeticError('Unresolved positive scattering weights')
   cost=kp[:,3]/kp[:,0];rate+=float(sum(weight));transport+=float(weight@(1-cost));dE+=float(weight@(kp[:,0]-Ein));outE+=float(weight@kp[:,0]);maxres=max(maxres,float(np.max(abs(k+pv-kp-pp))));hist+=np.histogram(kp[:,0],bins=edges,weights=weight)[0]
 return dict(rate=rate,transport_rate=transport,neutrino_energy_change=dE,outgoing_energy_moment=outE,target_density=density,energy_ledger=dE-(outE-Ein*rate),max_four_residual=maxres,T=T,mu_e=mu_e,Ein=Ein,flavor=coupling.flavor,positron=positron,anti=anti,final_neutrino_blocking='NONE'if radiation_mu is None else dict(T=T,mu=radiation_mu),histogram_unqualified=dict(edges=edges.tolist(),rate=hist.tolist()),controls=dict(radial=radial,incident=incident,outgoing=outgoing,phi=phi,tail=tail),seconds=time.monotonic()-start,units='rate s^-1; signedνenergychange MeV s^-1 per incidentν; targetdensity fm^-3')

def differential(T,mu_e,Ein,Eout,c,coupling,*,positron=False,anti=False,order=24,phi=8,tail=60.):
 """dΓ/(dEout dcos): exact targetenergy delta, no finalνblocking."""
 if not isinstance(positron,bool)or not isinstance(anti,bool)or not isinstance(coupling,P.Couplings):raise ValueError('Named coupling and literal charge flags required')
 if not all(math.isfinite(z)for z in [T,mu_e,Ein,Eout,c,tail])or min(T,Ein,Eout)<=0 or not -1<c<1 or not 20<=tail<=100:raise ValueError('Finite positive selectedenergies and interiorangle')
 for n in [order,phi]:
  if isinstance(n,bool)or not isinstance(n,int)or n<4:raise ValueError('Integer quadrature>=4')
 if not .2<=T<=4 or abs(mu_e/T)>3 or max(Ein,Eout)>150:raise ValueError('Selected warm pair-rich domain')
 w=Ein-Eout;q=math.sqrt(Ein**2+Eout**2-2*Ein*Eout*c);a=q*q-w*w;emin=(-w+q*math.sqrt(1+4*ME*ME/a))/2;mu=-mu_e if positron else mu_e;upper=max(emin,mu,mu-w)+tail*T;cuts=sorted(set([emin,upper]+[z for center in [mu,mu-w]for h in [-8,-3,0,3,8]if emin<(z:=center+h*T)<upper]));x,g=leggauss(order);ph=2*math.pi*np.arange(phi)/phi;total=0.;reverse_same=0.;kz=(Ein*Ein+q*q-Eout*Eout)/(2*q);kx=math.sqrt(max(0.,Ein*Ein-kz*kz));k0=np.array([Ein,kx,0,kz]);kp0=np.array([Eout,kx,0,kz-q])
 for lo,hi in zip(cuts[:-1],cuts[1:]):
  ee=(lo+hi)/2+(hi-lo)*x/2;EE,PH=np.meshgrid(ee,ph,indexing='ij');ee=EE.ravel();phv=PH.ravel();ww=np.repeat(g*(hi-lo)/2,phi)/phi;pz=(2*ee*w+w*w-q*q)/(2*q);r2=ee*ee-ME*ME-pz*pz
  if np.min(r2)<-1e-8:raise ArithmeticError('Target shell supportlost')
  rr=np.sqrt(np.maximum(0.,r2));pv=np.column_stack([ee,rr*np.cos(phv),rr*np.sin(phv),pz]);pp=pv+np.array([w,0,0,q]);k=np.tile(k0,(len(ee),1));kp=np.tile(kp0,(len(ee),1));amp=amplitude(k,pv,kp,pp,coupling,positron=positron,anti=anti);base=ww*amp;total+=float(base@(expit((mu-ee)/T)*expit((ee+w-mu)/T)));reverse_same+=float(base@(expit((mu-ee-w)/T)*expit((ee-mu)/T)))
 pref=Eout/(64*math.pi**3*Ein*q*HBAR)
 return dict(value=pref*total,reverse_same_phase=pref*reverse_same,emin=emin,upper=upper,Ein=Ein,Eout=Eout,cosine=c,controls=dict(order=order,phi=phi,tail=tail),units='s^-1 MeV^-1 per dcos; no finalνblocking')
