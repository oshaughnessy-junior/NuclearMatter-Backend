"""Independent Dirac trace, analytic azimuth and adaptive transverse shell."""
import sys,math,json,hashlib,importlib.util,time
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import expit
from numpy.polynomial.legendre import leggauss
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).parent;LEAF=ROOT/'neutrino_lepton_scattering';sys.dont_write_bytecode=True
before={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for d in ['neutrino_lepton_scattering','leptonic_pair_annihilation','relativistic_capture_oracle']for p in(ROOT/d).rglob('*')if p.is_file()and'__pycache__'not in str(p)}
spec=importlib.util.spec_from_file_location('review_frozen_lepton_scattering',LEAF/'scatter.py');S=importlib.util.module_from_spec(spec);sys.modules[spec.name]=S;spec.loader.exec_module(S)
GF=1.1663785e-11;M=.51099895069;HBAR=6.582119569e-22;MET=np.array([1,-1,-1,-1]);checks=[]
def ck(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and abs(error)<=tol)))
# Own gamma matrix definition; no author's gamma or spinor routine.
sig=[np.array([[0,1],[1,0]],complex),np.array([[0,-1j],[1j,0]],complex),np.diag([1,-1])];Z=np.zeros((2,2));I2=np.eye(2);I4=np.eye(4);G=np.array([np.diag([1,1,-1,-1])]+[np.block([[Z,s],[-s,Z]])for s in sig]);G5=1j*G[0]@G[1]@G[2]@G[3]
def slash(p):return np.einsum('i,i,ijk->jk',p,MET,G)
def adj(a):return G[0]@a.conj().T@G[0]
def traceamp(k,p,kp,pp,v,a,pos,anti):
 V=[g@(I4-G5)for g in G];C=[g@(v*I4-a*G5)for g in G]
 n1,n2=(k,kp)if anti else(kp,k);e1,e2=(p,pp)if pos else(pp,p);mass=-M if pos else M
 L=np.array([[np.trace(slash(n1)@V[i]@slash(n2)@adj(V[j]))for j in range(4)]for i in range(4)]);H=np.array([[np.trace((slash(e1)+mass*I4)@C[i]@(slash(e2)+mass*I4)@adj(C[j]))for j in range(4)]for i in range(4)])
 z=GF*GF/4*np.einsum('i,j,ij,ij',MET,MET,L,H)
 ck('independent trace real',z.imag/max(abs(z.real),1e-300),1e-12);return z.real
for p in [.2,3.7,14.]:
 for c in [-.67,.29]:
  E=math.hypot(M,p);k=np.array([p,0,0,p]);e=np.array([E,0,0,-p]);kp=np.array([p,p*math.sqrt(1-c*c),0,p*c]);ep=k+e-kp
  for flavor in ['e','mu']:
   C=S.P.Couplings(flavor)
   for pos in [False,True]:
    for anti in [False,True]:
     own=traceamp(k,e,kp,ep,C.cv,C.ca,pos,anti);actual=S.amplitude(k[None],e[None],kp[None],ep[None],C,positron=pos,anti=anti)[0]
     ck(f'fresh literal {p,c,flavor,pos,anti}',own/actual-1,2e-12)
# Independent analytic phi average, integrate transverse target momentum r.
def differential(T,mu_e,E,F,c,C,pos=False,anti=False,tail=70.):
 om=E-F;Q=math.sqrt((E-F)**2+2*E*F*(1-c));a=2*E*F*(1-c);emin=(-om+Q*math.sqrt(1+4*M*M/a))/2;mu=-mu_e if pos else mu_e;upper=max(emin,mu,mu-om)+tail*T
 kz=(E*E+Q*Q-F*F)/(2*Q);kx2=max(0.,E*E-kz*kz);h=E*F*(1-c);L,R=(C.cv+C.ca)**2,(C.cv-C.ca)**2
 if pos!=anti:L,R=R,L
 pz=lambda e:(2*e*om-a)/(2*Q)
 rmax=math.sqrt(max(0.,upper*upper-M*M-pz(upper)**2))
 def integrand(r):
  root=math.sqrt(1+4*(M*M+r*r)/a);et=(-om+Q*root)/2;der=2*Q*r/(a*root);u=E*et-kz*pz(et);vv=kx2*r*r/2
  amp=L*(u*u+vv)+R*((u-h)**2+vv)-M*M*(C.cv*C.cv-C.ca*C.ca)*h
  return der*amp*expit((mu-et)/T)*expit((et+om-mu)/T)
 value,err=quad(integrand,0,rmax,epsabs=2e-9,epsrel=2e-10,limit=120)
 return F/(64*math.pi**3*E*Q*HBAR)*16*GF*GF*value
T=1.4;mu=-.11;E=6.7;C=S.P.Couplings('mu');pos=anti=True
points=[]
for F,c in [(3.9,-.38),(10.2,.46),(6.85,.97)]:
 own=differential(T,mu,E,F,c,C,pos,anti);auth=S.differential(T,mu,E,F,c,C,positron=pos,anti=anti,order=36,phi=8,tail=80.)['value'];rev=differential(T,mu,F,E,c,C,pos,anti)
 ck('heldout transverse-shell absolute '+str((F,c)),own/auth-1,3e-8)
 ck('DOS KMS '+str((F,c)),rev/own/((E/F)**2*math.exp((F-E)/T))-1,2e-9)
 # arbitrary commonT muν stationary after one-helicity DOS and BOTH FD factors.
 for munu in [-.7,2.1]:
  f=expit((munu-E)/T);fp=expit((munu-F)/T);left=E*E*f*(1-fp)*own;right=F*F*fp*(1-f)*rev;ck('local FD stationarity '+str((F,c,munu)),left/right-1,2e-9)
 points.append(dict(Eout=F,c=c,independent=own,author=auth,reverse=rev))
def moments(order,angle,upper,tail=70.):
 start=time.monotonic();xx,ww=leggauss(order);zz,zw=leggauss(angle);vals=np.zeros(4)
 # z=sin(theta/2); dc=4z dz. Independent analytic phi+adaptive r inner.
 for z,wz in zip((zz+1)/2,zw/2):
  c=1-2*z*z;width=2*E*z;edges=sorted(set([0.,upper,1.,3.,E,12.,22.,40.,70.]+[E+h*width for h in [-4,-2,-1,-.5,-.2,.2,.5,1,2,4]if 0<E+h*width<upper]));edges=[v for v in edges if v<=upper]
  for lo,hi in zip(edges[:-1],edges[1:]):
   for f,wf in zip((lo+hi)/2+xx*(hi-lo)/2,ww*(hi-lo)/2):
    if time.monotonic()-start>110:raise TimeoutError('Bounded independent shell moments')
    d=differential(T,mu,E,float(f),float(c),C,pos,anti,tail);v=d*wf*wz*4*z;vals+=v*np.array([1,1-c,f-E,f])
 return dict(values=vals.tolist(),order=order,angle=angle,upper=upper,tail=tail,seconds=time.monotonic()-start)
coarse=moments(12,16,100.);fine=moments(18,24,150.);cm=S.thermal(T,mu,E,C,positron=pos,anti=anti,radial=28,incident=36,outgoing=36,phi=44,tail=75.,budget=110.)
for i,key in enumerate(['rate','transport_rate','neutrino_energy_change','outgoing_energy_moment']):
 ck('heldout shell refinement '+key,coarse['values'][i]/fine['values'][i]-1,3e-6)
 ck('heldout independent shell vs CM '+key,fine['values'][i]/cm[key]-1,3e-6)
tail=moments(18,24,150.,tail=90.)
for i,key in enumerate(['rate','transport_rate','neutrino_energy_change','outgoing_energy_moment']):ck('independent transverse parent tail70/90 '+key,fine['values'][i]/tail['values'][i]-1,3e-8)
mu_target=-mu
density=quad(lambda p:p*p*expit((mu_target-math.hypot(p,M))/T),0,200.,epsabs=1e-11,epsrel=2e-12)[0]/(math.pi**2*197.3269804**3)
ck('heldout adaptive actual target density',cm['target_density']/density-1,2e-9)
ck('signed neutrino/material energy bookkeeping',cm['neutrino_energy_change']-(cm['outgoing_energy_moment']-E*cm['rate']),1e-10)
old=json.loads((LEAF/'results.json').read_text());ck('six coarse author failures preserved',len([x for x in old['checks']if not x['passed']])-6,0)
after={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()for name in before};ck('all frozen source/raw unchanged',float(before!=after),0)
out=dict(checks=checks,passed=sum(x['passed']for x in checks),failed=[x for x in checks if not x['passed']],state=dict(T=T,mu_e=mu,Ein=E,flavor='mu',positron=pos,anti=anti),points=points,coarse=coarse,fine=fine,tail=tail,independent_target_density=density,cm=cm,frozen_before=before,frozen_after=after)
(HERE/'results.json').write_text(json.dumps(out,indent=2));print(json.dumps(dict(passed=out['passed'],failed=out['failed'],coarse=coarse,fine=fine,cm=cm),indent=2))
