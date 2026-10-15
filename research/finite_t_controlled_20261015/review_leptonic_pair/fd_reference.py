"""Independent p=T sinh(t), outgoing BOOST-axis CM integration; source scalar current."""
import math,time
import numpy as np
from scipy.special import log_expit
from numpy.polynomial.legendre import leggauss
from reference import GF,HC,HBAR,ME
MET=np.array([1.,-1.,-1.,-1.])
def dot(a,b):return np.sum(a*b*MET,axis=1)
def boost(p,beta):
 b2=np.sum(beta*beta,axis=1);g=1/np.sqrt(1-b2);d=np.sum(p[:,1:]*beta,axis=1)
 return np.column_stack([g*(p[:,0]+d),p[:,1:]+(g*p[:,0]+g*g/(g+1)*d)[:,None]*beta])
def integrate(T,mu,v,a,scenarios,order=28,angle=8,phi=8,tail=40.,budget=110.):
 start=time.monotonic();x,w=leggauss(order);ci,wi=leggauss(angle);co,wo=leggauss(angle);ph=2*math.pi*np.arange(phi)/phi;end=math.asinh((abs(mu)+tail*T)/T);t=(x+1)*end/2;p=T*np.sinh(t);E=np.hypot(ME,p);pw=w*end/2*p*p*T*np.cosh(t)
 PP,CI,CO,PH=np.meshgrid(p,ci,co,ph,indexing='ij');pp=PP.ravel();ee=np.hypot(ME,pp);ci=CI.ravel();co=CO.ravel();ph=PH.ravel();aw=(pw[:,None,None,None]*wi[None,:,None,None]*wo[None,None,:,None]*np.ones((1,1,1,phi))*2*math.pi/phi).ravel();N=len(pp);out={name:dict(F=np.zeros(5),B=np.zeros(5),net=np.zeros(5),entropy=0.,log_ratio_error=0.)for name in scenarios};density=np.zeros(2)
 for e,pe,wm in zip(E,p,pw):
  if time.monotonic()-start>budget:raise TimeoutError('Independent pair review budget')
  density[0]+=wm/math.pi**2/HC**3*math.exp(float(log_expit((mu-e)/T)));density[1]+=wm/math.pi**2/HC**3*math.exp(float(log_expit((-mu-e)/T)))
  pm=np.tile([e,0,0,pe],(N,1));pv=np.column_stack([ee,pp*np.sqrt(1-ci*ci),np.zeros(N),pp*ci]);total=pm+pv;s=total[:,0]**2-np.sum(total[:,1:]**2,axis=1);rt=np.sqrt(s);beta=total[:,1:]/total[:,0,None];speed=np.linalg.norm(beta,axis=1);axis=beta/speed[:,None];y=np.tile([0,1,0],(N,1));xx=np.cross(y,axis);direc=co[:,None]*axis+np.sqrt(1-co*co)[:,None]*(np.cos(ph)[:,None]*xx+np.sin(ph)[:,None]*y)
  k=boost(np.column_stack([rt/2,rt[:,None]/2*direc]),beta);anti=boost(np.column_stack([rt/2,-rt[:,None]/2*direc]),beta)
  amp=8*GF*GF*((a-v)**2*dot(pm,k)*dot(pv,anti)+(a+v)**2*dot(pv,k)*dot(pm,anti)+ME*ME*(v*v-a*a)*dot(k,anti));weight=wm/(2*math.pi**4*HBAR*HC**3)*aw*amp/(128*math.pi**2*e*ee)
  if np.any(weight<=0)or np.any(~np.isfinite(weight)):raise ArithmeticError('Unresolved independent positive event measure')
  fm=log_expit((mu-e)/T);fp=log_expit((-mu-ee)/T);hm=log_expit((e-mu)/T);hp=log_expit((ee+mu)/T);Energies=np.column_stack([np.ones(N),k[:,0],anti[:,0],np.full(N,e),ee]);base=np.log(weight)
  for name,rad in scenarios.items():
   acc=out[name]
   if rad is None:lf=fm+fp;lb=None
   else:
    un,ua=rad;fn=log_expit((un-k[:,0])/T);hn=log_expit((k[:,0]-un)/T);fa=log_expit((ua-anti[:,0])/T);ha=log_expit((anti[:,0]-ua)/T);lf=fm+fp+hn+ha;lb=hm+hp+fn+fa;delta=lf-lb;acc['log_ratio_error']=max(acc['log_ratio_error'],float(np.max(abs(delta+(un+ua)/T))))
   F=np.exp(base+lf);acc['F']+=F@Energies
   if lb is None:net=F
   else:
    B=np.exp(base+lb);acc['B']+=B@Energies;net=np.sign(delta)*np.exp(base+np.maximum(lf,lb))*(-np.expm1(-abs(delta)));acc['entropy']+=float(net@delta)
   acc['net']+=net@Energies
 return dict(T=T,mu=mu,density=density,scenarios=out,controls=dict(order=order,angle=angle,phi=phi,tail=tail,radial_coordinate='p/T=sinh(t)',outgoing_axis='CMboostdirection'),seconds=time.monotonic()-start)
