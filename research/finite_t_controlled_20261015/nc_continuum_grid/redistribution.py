"""Continuum and sampled-energy-grid NC moments in a fixed freeNR bath."""
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache
import sys,math
import numpy as np
from scipy.special import expit
from scipy.integrate import quad_vec
from numpy.polynomial.legendre import leggauss
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'neutral_current'))
import free_nc as nc
from free_density_response import log_response
nodes_cached=lru_cache(maxsize=32)(leggauss)
LABELS=['blocked_out_rate_s','blocked_transport_rate_s','energy_to_bath_MeV_s','absolute_energy_transfer_MeV_s','gain_s','loss_s','net_collision_s']
@dataclass(frozen=True)
class Radiation:
 kind:str='FD'
 T:float=3.
 mu:float=0.
 def __post_init__(self):
  if self.kind not in ['FD','packet']or not math.isfinite(self.T)or self.T<=0 or not math.isfinite(self.mu):raise ValueError('Finite positive radiation T and named FD/packet distribution required')
 def f(self,E):
  base=float(expit((self.mu-E)/self.T))
  return base if self.kind=='FD'else .15*base+.45*math.exp(-((E-14.)/3.)**2)

def controls(E,order):
 if not math.isfinite(E)or E<=0:raise ValueError('Finite positive incident energy required')
 if isinstance(order,bool)or not isinstance(order,int)or order<8:raise ValueError('Integer order>=8 required')

def upper(species,E,tail=50.):
 if not math.isfinite(tail)or not 30<=tail<=100:raise ValueError('Tail30..100T required')
 q=2*E;width=max(q*math.sqrt(species.T/species.mass),q*math.sqrt(2*max(species.kinetic_mu,0.)/species.mass),q*q/(2*species.mass))
 return E+max(tail*species.T,24*width)

def edges(species,E,cutoff):
 width=max(E*math.sqrt(species.T/species.mass),E*math.sqrt(2*max(species.kinetic_mu,0.)/species.mass),E*E/(2*species.mass))
 values=[0.,cutoff,E]
 values.extend(E+k*width for k in [-24,-12,-6,-3,-1,-.3,.3,1,3,6,12,24]if 0<E+k*width<cutoff)
 return sorted(set(values))

def integrand(species,E,Ep,radiation,*,angle_order=48,coordinate='q'):
 controls(E,angle_order)
 if not math.isfinite(Ep)or Ep<0:raise ValueError('Finite outgoing energy>=0 required')
 if Ep==0:return np.zeros(7)
 if coordinate not in ['q','cos']:raise ValueError('q or cos coordinate required')
 pair=nc.pair_from_species(species);xs,ws=nodes_cached(angle_order)
 if coordinate=='q':
  lo,hi=abs(E-Ep),E+Ep
  if hi<=lo:raise ArithmeticError('q interval collapsed')
  qs=lo+(xs+1)*(hi-lo)/2;cs=(E*E+Ep*Ep-qs*qs)/(2*E*Ep);weights=ws*(hi-lo)/2*qs/(E*Ep)
 else:
  cs=xs;weights=ws;qs=np.sqrt((E-Ep)**2+2*E*Ep*(1-cs))
 fi,fo=radiation.f(E),radiation.f(Ep);omega=E-Ep
 logs=[];trans=[]
 for q,c,w in zip(qs,cs,weights):
  c=max(-1.,min(1.,float(c)))
  logs.append(math.log(nc.K)+2*math.log(Ep)+math.log(nc.angular_weight(c,nc.NEUTRON))+log_response(pair,float(q),omega))
  trans.append(1-c)
 kernel=np.exp(logs);rate=float(weights@kernel);transport=float(weights@(kernel*np.array(trans)))
 gain=rate*math.exp(-omega/species.T)*fo*(1-fi);loss=rate*fi*(1-fo)
 if radiation.kind=='FD':delta=omega*(1/radiation.T-1/species.T)
 else:
  delta=math.log(fo)+math.log1p(-fi)-math.log(fi)-math.log1p(-fo)-omega/species.T
 # Stable cancellation only; this does not alter S or impose equilibrium on f.
 net=loss*math.expm1(delta)
 return np.array([rate*(1-fo),transport*(1-fo),omega*rate*(1-fo),abs(omega)*rate*(1-fo),gain,loss,net])

def adaptive(species,E,radiation,*,angle_order=48,coordinate='q',tail=50.,rtol=2e-10):
 controls(E,angle_order)
 if not math.isfinite(rtol)or not 1e-12<=rtol<=1e-6:raise ValueError('rtol1e-12..1e-6 required')
 end=upper(species,E,tail);es=edges(species,E,end);total=np.zeros(7);errs=0.
 for lo,hi in zip(es[:-1],es[1:]):
  val,err,info=quad_vec(lambda x:integrand(species,E,float(x),radiation,angle_order=angle_order,coordinate=coordinate),lo,hi,epsabs=1e-14,epsrel=rtol,limit=250,full_output=True)
  if not info.success and not(info.status==2 and err<=max(1e-14,rtol*np.max(abs(val)))):raise ArithmeticError(info.message)
  total+=val;errs+=err
 return dict(values=total.tolist(),estimated_absolute_vector_error=float(errs),angle_order=angle_order,coordinate=coordinate,tail=tail,cutoff=end,rtol=rtol)

def energy_grid(species,E,radiation,*,kind='resolved',order=16,angle_order=64,tail=70.):
 controls(E,order);end=upper(species,E,tail)
 if kind=='uniform':
  es=np.linspace(0,end,order+1);nodes=.5*(es[:-1]+es[1:]);weights=np.diff(es)
 elif kind=='resolved':
  xs,ws=nodes_cached(order);nodes=[];weights=[]
  es=edges(species,E,end)
  for lo,hi in zip(es[:-1],es[1:]):nodes.extend(lo+(xs+1)*(hi-lo)/2);weights.extend(ws*(hi-lo)/2)
  nodes=np.asarray(nodes);weights=np.asarray(weights)
 else:raise ValueError('Uniform or resolved grid required')
 total=np.zeros(7)
 for node,weight in zip(nodes,weights):total+=float(weight)*integrand(species,E,float(node),radiation,angle_order=angle_order,coordinate='q')
 return dict(values=total.tolist(),kind=kind,order=order,nodes=nodes.tolist(),weights=weights.tolist(),angle_order=angle_order,tail=tail,cutoff=end)
