"""New spacelike external-spinor and timelike/null delta-shell tensor witnesses."""
from pathlib import Path
import sys,math,json,time,hashlib,cmath,atexit
import numpy as np
from scipy.special import expit
from numpy.polynomial.legendre import leggauss
P=Path(__file__).resolve().parent;ROOT=P.parent;sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'relativistic_tensor'));import tensor as t
sys.path.insert(0,str(ROOT/'relativistic_delta_oracle'));import oracle as o
paths=[p for folder in ('relativistic_tensor','relativistic_delta_oracle')for p in (ROOT/folder).iterdir()if p.is_file()]
sha=lambda:{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths}
before=sha();start=time.monotonic();checks=[];rows=[]
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and error<tol)))
def serialize(x):
 if isinstance(x,np.ndarray):return dict(real=x.real.tolist(),imag=x.imag.tolist())if np.iscomplexobj(x)else x.tolist()
 if isinstance(x,np.generic):return x.item()
 raise TypeError(type(x).__name__)
def save():
 (P/'results.json').write_text(json.dumps(dict(checks=checks,rows=rows,source_hashes_before=before,source_hashes_after=sha(),runtime_seconds=time.monotonic()-start),indent=2,default=serialize)+'\n')
atexit.register(save)
# New spacelike bath/lepton points, shared only exact physical inputs.
for p in [o.Point(7.3,939.56542052,938.27208816,948.2,940.1,97.,82.,-.17,105.6583755 if False else .51099895069,31.2),o.Point(16.,938.27208816,939.56542052,943.,954.,191.,142.,.31,105.6583755,-24.,True)]:
 state=t.State(p.T,p.ma,p.mb,p.mua,p.mub,p.Mweak);a=t.differential(state,p.Enu,p.El,p.cosine,p.ml,p.mul,antineutrino=p.anti,energy_order=40,phi_order=16,tail=80.);b=o.integrate(p,order=40,nphi=16,tail=80.)
 rows.append(dict(kind='fresh_spacelike',point=vars(p),author=a,oracle=b))
 for key in ('j','chi'):gate('freshspacelike anti'+str(p.anti)+key,abs(a[key]['value']/b[key]-1),3e-8)
 for mode,key in enumerate(('absorption','emission')):
  X=a['response'][key]['tensor'];Y=b['W'][mode]
  for i in range(4):
   for j in range(4):gate(f'freshspacelike{p.anti}W{mode}{i}{j}',abs(X[i,j]-Y[i,j])/np.max(abs(Y)),3e-8)
 gate('freshspacelike L all16 '+str(p.anti),np.max(abs(a['lepton_tensor']-b['lepton_tensor']))/np.max(abs(b['lepton_tensor'])),3e-12)
 gate('freshspacelike independent KMS '+str(p.anti),abs(math.log(b['j']/b['chi'])+(p.Enu-(p.mul+p.mub-p.mua))/p.T),3e-12)
 save()

# Timelike transfer: integrate a fixed-energy sphere in the q-rest frame,
# boost each external spinor to the bath frame and evaluate original delta J.
def timelike(sp,w,q,n=32,phi_n=16):
 ma,mb=sp.mass_a,sp.mass_b;B=w*w-q*q;root=math.sqrt(B);eta=1 if w>0 else -1;D=mb*mb-ma*ma-B;E0=D/(2*eta*root);R=math.sqrt((E0-ma)*(E0+ma));xs,ws=leggauss(n);V=o.currents(np.array([w,0.,0.,q]),sp.weak_mass);W=np.zeros((2,4,4),complex);jac_errors=[]
 for x,weight in zip(xs,ws):
  zp=R*x;r=R*math.sqrt(1-x*x);Ea=(abs(w)*E0+eta*q*zp)/root;z=(eta*q*E0+abs(w)*zp)/root;Eb=Ea+w
  # Independent original delta check; no author support polynomial used.
  original=math.sqrt(mb*mb+r*r+(z+q)**2)-math.sqrt(ma*ma+r*r+z*z)-w
  jac=abs((z+q)/Eb-z/Ea);h=.003;dz=cmath.sqrt(mb*mb+r*r+(z+1j*h+q)**2)-cmath.sqrt(ma*ma+r*r+(z+1j*h)**2)-w
  jac_errors.append(abs(abs(dz.imag/h)/jac-1));gate('timelike original delta',abs(original),3e-10)
  occ=np.array([expit((sp.mu_a-Ea)/sp.T)*expit((Eb-sp.mu_b)/sp.T),expit((sp.mu_b-Eb)/sp.T)*expit((Ea-sp.mu_a)/sp.T)])
  for phi in 2*math.pi*np.arange(phi_n)/phi_n:
   pa=np.array([Ea,r*math.cos(phi),r*math.sin(phi),z]);pb=pa+np.array([w,0.,0.,q]);ua=o.spinors(pa,ma);ub=o.spinors(pb,mb);J=np.array([(ub.conj().T@o.G[0]@v@ua).reshape(-1)for v in V]);H=J@J.conj().T
   W+=weight*(2*math.pi/phi_n)*R/(16*math.pi**2*root)*occ[:,None,None]*H[None,:,:]
 gate('timelike complex-step original Jacobian',max(jac_errors),3e-7)
 return W

def lightlike(sp,w,q,n=40,phi_n=16,tail=80.):
 ma,mb=sp.mass_a,sp.mass_b;eta=1 if w>0 else -1;A=(mb*mb-ma*ma)/(2*w)
 if A<=0:raise ValueError('No positive-energy null shell')
 E0=(A*A+ma*ma)/(2*A);end=max(E0,sp.mu_a,sp.mu_b-w)+tail*sp.T;umax=A/ma*(end-E0)
 cuts=[0.,umax]
 for center in (sp.mu_a,sp.mu_b-w):
  for h in (-16,-8,-3,0,3,8,16):
   E=center+h*sp.T
   if E0<E<end:cuts.append(A/ma*(E-E0))
 cuts=sorted(set(cuts));xs,ws=leggauss(n);W=np.zeros((2,4,4),complex);V=o.currents(np.array([w,0.,0.,q]),sp.weak_mass);shellmax=jacmax=0.
 for lo,hi in zip(cuts[:-1],cuts[1:]):
  for x,weight in zip(xs,ws):
   u=lo+(x+1)*(hi-lo)/2;r=math.sqrt(2*ma*u);Ea=(A*A+ma*ma+r*r)/(2*A);z=eta*(ma*ma+r*r-A*A)/(2*A);Eb=Ea+w;jac=abs((z+q)/Eb-z/Ea)
   delta=math.sqrt(mb*mb+r*r+(z+q)**2)-math.sqrt(ma*ma+r*r+z*z)-w;shellmax=max(shellmax,abs(delta));h=.003;dz=cmath.sqrt(mb*mb+r*r+(z+1j*h+q)**2)-cmath.sqrt(ma*ma+r*r+(z+1j*h)**2)-w;jacmax=max(jacmax,abs(abs(dz.imag/h)/jac-1))
   occ=np.array([expit((sp.mu_a-Ea)/sp.T)*expit((Eb-sp.mu_b)/sp.T),expit((sp.mu_b-Eb)/sp.T)*expit((Ea-sp.mu_a)/sp.T)])
   for phi in 2*math.pi*np.arange(phi_n)/phi_n:
    pa=np.array([Ea,r*math.cos(phi),r*math.sin(phi),z]);pb=pa+np.array([w,0.,0.,q]);ua=o.spinors(pa,ma);ub=o.spinors(pb,mb);J=np.array([(ub.conj().T@o.G[0]@v@ua).reshape(-1)for v in V]);H=J@J.conj().T
    W+=ma*weight*(hi-lo)/2*(2*math.pi/phi_n)/(16*math.pi**2*Ea*Eb*jac)*occ[:,None,None]*H[None,:,:]
 gate('lightlike original energy delta',shellmax,3e-9);gate('lightlike complex-step original Jacobian',jacmax,3e-7)
 return W
for label,sp,w,q in [('timelike',t.State(20.,810.,700.,845.,740.,938.91875434),-105.,20.),('lightlike',t.State(30.,939.56542052,938.27208816,960.,945.,938.91875434),-.7,.7)]:
 a=t.response(sp,w,q,energy_order=48,phi_order=16,tail=80.);b=timelike(sp,w,q,48,16)if label=='timelike'else lightlike(sp,w,q,48,16,80.)
 rows.append(dict(kind=label,state=vars(sp),omega=w,q=q,author=a,independent_W=b))
 for mode,key in enumerate(('absorption','emission')):
  X=a[key]['tensor'];Y=b[mode]
  for i in range(4):
   for j in range(4):gate(f'{label} allW{mode}{i}{j}',abs(X[i,j]-Y[i,j])/np.max(abs(Y)),3e-8)
  expected=-(w+sp.mu_a-sp.mu_b)/sp.T;gate(label+' independent occupation KMS '+key,np.max(abs(b[1]-math.exp(expected)*b[0]))/np.max(abs(b[1])),3e-11)
 save()
gate('all author files/failures unchanged',float(sha()!=before),.1);save();print(sum(c['passed']for c in checks),'/',len(checks),'runtime',time.monotonic()-start)
for c in checks:
 if not c['passed']:print('FAIL',c)
