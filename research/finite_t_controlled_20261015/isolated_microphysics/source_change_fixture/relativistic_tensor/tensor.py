"""Free relativistic particle-hole CC tensor, all16 complex components.

Full unaveraged baryon trace; W has MeV² units and no GF. Free U=0 only.
Independent emission/absorption occupations; explicit bare weak-current mass.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
from pathlib import Path
import sys,math
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import log_expit
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'muon_full_current'));import current as C
V=C.V;HBAR=6.582119569e-22

@dataclass(frozen=True)
class State:
 T:float
 mass_a:float
 mass_b:float
 mu_a:float
 mu_b:float
 weak_mass:float=(V.MN+V.MP)/2
 def __post_init__(self):
  if not all(math.isfinite(v)for v in asdict(self).values())or min(self.T,self.mass_a,self.mass_b,self.weak_mass)<=0:raise ValueError('Finite parameters and positive T/masses required')

def support(ma,mb,omega,q):
 """Exact positive-energy particle-hole support; upperNone means infinity."""
 if not all(math.isfinite(x)for x in(ma,mb,omega,q))or min(ma,mb,q)<=0:raise ValueError('Positive masses,q and finite transfer required')
 A=(q-abs(omega))*(q+abs(omega));Dmass=(ma-mb)*(ma+mb);cc=Dmass-A
 B=-omega*cc;D=-(cc*cc/4+q*q*ma*ma);low=max(ma,mb-omega)
 if A==0:
  if B==0:return [dict(lower=low,upper=None)]if D>=0 else[]
  edge=-D/B
  return [dict(lower=max(low,edge),upper=None)]if B>0 else([dict(lower=low,upper=edge)]if edge>low else[])
 disc=q*q*(cc*cc+4*A*ma*ma)
 if disc<0:return []
 root=math.sqrt(disc)
 stable=(-B-math.copysign(root,B))/(2*A)
 other=D/(A*stable)if stable else(-B+root)/(2*A)
 r1,r2=sorted((stable,other))
 if A<0:return [dict(lower=max(low,r1),upper=r2)]if r2>max(low,r1)else[]
 out=[]
 if r1>low:out.append(dict(lower=low,upper=r1))
 out.append(dict(lower=max(low,r2),upper=None))
 return out

def _trace(pa,pb,ma,mb,J,barJ):
 a=V.slash(pa)+ma*V.I;b=V.slash(pb)+mb*V.I
 left=np.array([b@j@a for j in J])
 return np.einsum('mij,nji->mn',left,barJ)

def _representation(scaled,logscale):
 if logscale is None:return dict(tensor=np.zeros((4,4),complex),representable=True,reason='empty_support')
 # Combine each component with the scale before exponentiating. A scale alone
 # may underflow even when its product with a large spin trace is represented.
 value=np.zeros_like(scaled);under=False;over=False
 for index in np.ndindex(scaled.shape):
  parts=[]
  for part in(scaled[index].real,scaled[index].imag):
   if part==0:parts.append(0.);continue
   try:v=math.exp(math.log(abs(float(part)))+logscale)
   except OverflowError:over=True;v=0.
   if v==0:under=True
   parts.append(math.copysign(v,part))
  value[index]=complex(*parts)
 if over:return dict(tensor=None,representable=False,reason='overflow')
 if under:return dict(tensor=None,representable=False,reason='component_underflow')
 return dict(tensor=value,representable=True,reason=None)

def response(state,omega,q,*,model=C.FormFactors(),energy_order=24,phi_order=8,tail=50.):
 if isinstance(energy_order,bool)or not isinstance(energy_order,int)or energy_order<8:raise ValueError('Energy integer order>=8 required')
 if isinstance(phi_order,bool)or not isinstance(phi_order,int)or phi_order<4:raise ValueError('Periodic phi integer order>=4 required')
 if not math.isfinite(tail)or not 30<=tail<=120:raise ValueError('Tail30..120 required')
 intervals=support(state.mass_a,state.mass_b,omega,q);q4=np.array([omega,0.,0.,q]);ff=model.values(omega*omega-q*q,state.weak_mass)
 pieces=C.vertices(q4,state.weak_mass,ff);J=np.array([sum(pieces[key][i]for key in pieces)for i in range(4)]);barJ=np.array([V.adjoint(j)for j in J])
 xs,ws=leggauss(energy_order);nodes=[];panels=[]
 for interval in intervals:
  lower=interval['lower'];upper=max(lower,state.mass_a,state.mu_a,state.mu_b-omega)+tail*state.T
  if interval['upper']is not None:upper=min(upper,interval['upper'])
  if upper<=lower:continue
  cuts=sorted({lower,upper}|{x for center in(state.mu_a,state.mu_b-omega)for k in(-30,-12,-4,0,4,12,30)if lower<(x:=center+k*state.T)<upper})
  panels.extend([dict(lower=a,upper=b)for a,b in zip(cuts[:-1],cuts[1:])])
  for a,b in zip(cuts[:-1],cuts[1:]):
   for x,w in zip(xs,ws):nodes.append((float(a+(x+1)*(b-a)/2),float(w*(b-a)/2)))
 if not nodes:
  zero=np.zeros((4,4),complex)
  return dict(absorption=dict(scaled=zero,log_scale=None,**_representation(zero,None)),emission=dict(scaled=zero.copy(),log_scale=None,**_representation(zero,None)),support=intervals,panels=panels,controls=dict(energy_order=energy_order,phi_order=phi_order,tail=tail),max_angle_roundoff=0.,state=asdict(state),omega=omega,q=q,form_factors=ff)
 logs=np.array([[float(log_expit((state.mu_a-Ea)/state.T)+log_expit((Ea+omega-state.mu_b)/state.T)),float(log_expit((state.mu_b-Ea-omega)/state.T)+log_expit((Ea-state.mu_a)/state.T))]for Ea,w in nodes])
 if not np.all(np.isfinite(logs)):raise ArithmeticError('Occupation logarithms outside finite represented domain')
 scales=np.max(logs,axis=0);sums=np.zeros((2,4,4),complex);roundoff=0.
 for index,(Ea,weight)in enumerate(nodes):
  p=math.sqrt((Ea-state.mass_a)*(Ea+state.mass_a));Eb=Ea+omega
  numerator=(state.mass_a-state.mass_b)*(state.mass_a+state.mass_b)+omega*omega-q*q+2*omega*Ea
  cosine=numerator/(2*p*q);excess=max(0.,abs(cosine)-1);roundoff=max(roundoff,excess)
  if excess>1e-10:raise ArithmeticError('Support angle outside domain')
  cosine=max(-1.,min(1.,cosine));pt=p*math.sqrt(max(0.,1-cosine*cosine));pz=p*cosine
  average=np.zeros((4,4),complex)
  for j in range(phi_order):
   phi=2*math.pi*j/phi_order;px,py=pt*math.cos(phi),pt*math.sin(phi)
   pa=np.array([Ea,px,py,pz]);pb=np.array([Eb,px,py,pz+q]);average+=_trace(pa,pb,state.mass_a,state.mass_b,J,barJ)/phi_order
  for mode in range(2):sums[mode]+=weight/(8*math.pi*q)*math.exp(logs[index,mode]-scales[mode])*average
 return dict(absorption=dict(scaled=sums[0],log_scale=float(scales[0]),**_representation(sums[0],float(scales[0]))),emission=dict(scaled=sums[1],log_scale=float(scales[1]),**_representation(sums[1],float(scales[1]))),support=intervals,panels=panels,controls=dict(energy_order=energy_order,phi_order=phi_order,tail=tail),max_angle_roundoff=roundoff,state=asdict(state),omega=omega,q=q,form_factors=ff)

def _coefficient(branch,L,loglepton,pref):
 if branch['log_scale']is None:return dict(value=0.,log=None,representable=True,reason='empty_support')
 z=np.sum(V.METRIC[:,None]*V.METRIC[None,:]*L*branch['scaled'])
 if abs(z.imag)>1e-10*max(abs(z.real),1e-300)or z.real<=0:raise ArithmeticError('Nonpositive/nonreal physical tensor contraction')
 log=math.log(pref)+math.log(float(z.real))+branch['log_scale']+float(loglepton)
 try:value=math.exp(log)
 except OverflowError:return dict(value=None,log=log,representable=False,reason='overflow')
 return dict(value=value if value else None,log=log,representable=value>0,reason=None if value else'underflow')

def differential(state,Enu,El,cosine,lepton_mass,mu_lepton,*,antineutrino=False,model=C.FormFactors(),**controls):
 if not all(math.isfinite(v)for v in(Enu,El,cosine,lepton_mass,mu_lepton))or Enu<=0 or lepton_mass<0 or El<=lepton_mass or not -1<=cosine<=1:raise ValueError('Valid positive energies, El>mass and cosine required')
 if not isinstance(antineutrino,bool):raise ValueError('Explicit antineutrino bool required')
 pl=math.sqrt((El-lepton_mass)*(El+lepton_mass));omega=Enu-El;q=math.sqrt((Enu-pl)**2+2*Enu*pl*(1-cosine))
 r=response(state,omega,q,model=model,**controls);kz=(Enu*Enu+q*q-pl*pl)/(2*q);kx2=Enu*Enu-kz*kz
 if kx2< -1e-10*Enu*Enu:raise ArithmeticError('Lepton q-frame outside mass shell')
 kx=math.sqrt(max(0.,kx2));k=np.array([Enu,kx,0.,kz]);l=np.array([El,kx,0.,kz-q])
 left=[g@(V.I-V.G5)for g in V.G];spin=V.slash(l)+(-1 if antineutrino else 1)*lepton_mass*V.I
 L=np.array([[np.trace(spin@left[u]@V.slash(k)@V.adjoint(left[v]))for v in range(4)]for u in range(4)])
 if antineutrino:L=L.T
 pref=V.GF**2*V.VUD**2/(32*math.pi**2)*pl/Enu/HBAR
 chi=_coefficient(r['absorption'],L,log_expit((El-mu_lepton)/state.T),pref)
 j=_coefficient(r['emission'],L,log_expit((mu_lepton-El)/state.T),pref)
 return dict(j=j,chi=chi,response=r,lepton_tensor=L,lepton_k=k,lepton_l=l,inputs=dict(Enu=Enu,El=El,cosine=cosine,lepton_mass=lepton_mass,mu_lepton=mu_lepton,antineutrino=antineutrino),units='differential j/chi: s^-1 MeV^-1 per dcos; W: MeV²')
