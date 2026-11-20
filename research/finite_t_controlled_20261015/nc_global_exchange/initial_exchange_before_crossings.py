"""One-helicity free-NR NC double-energy exchange in an explicit finite box."""
from pathlib import Path
import sys,math,hashlib,platform,time
import numpy as np
from scipy.special import expit,log_expit
from numpy.polynomial.legendre import leggauss
from functools import lru_cache
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'nc_continuum_grid'));import redistribution as reference
nc=reference.nc
nodes=lru_cache(maxsize=24)(leggauss)
SOURCE_FILES=('neutral_current/free_nc.py','nr_baryon_eos/free_nucleons.py','response_reference/free_density_response.py','thermodynamics/lepton_reference.py','weak_rates/static_capture.py','allowed_opacity/allowed_capture.py','nc_continuum_grid/redistribution.py','nc_global_exchange/exchange.py')
def hashes():return {f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in SOURCE_FILES}
class Radiation:
 def __init__(self,T,mu,packet=False):
  if not all(math.isfinite(x) for x in (T,mu)) or T<=0 or not isinstance(packet,bool):raise ValueError('Finite T>0,mu and explicit bool packet required')
  self.T,self.mu,self.packet=T,mu,packet
 def logs(self,E):
  if not math.isfinite(E) or E<0:raise ValueError('Finite E>=0 required')
  x=(self.mu-E)/self.T
  if not self.packet:return float(log_expit(x)),float(log_expit(-x))
  lf=float(np.logaddexp(math.log(.15)+log_expit(x),math.log(.3)-((E-12.)/4.)**2))
  return lf,math.log1p(-math.exp(lf))
 def metadata(self):return dict(T=self.T,mu=self.mu,packet=self.packet)
def control(cutoff,energy_order,angle_order,inner_order):
 if not math.isfinite(cutoff) or cutoff<=0:raise ValueError('Finite positive box cutoff required')
 for order in (energy_order,angle_order,inner_order):
  if isinstance(order,bool) or not isinstance(order,int) or order<4:raise ValueError('Integer quadrature orders>=4 required')
def energy_edges(cut):return sorted(set([0.,cut]+[x for x in (2.,5.,10.,16.,24.,36.,52.,72.,100.) if x<cut]))
def partner_edges(sp,E,lo,hi):
 width=max(E*math.sqrt(sp.T/sp.mass),E*math.sqrt(2*max(sp.kinetic_mu,0)/sp.mass),E*E/(2*sp.mass))
 return sorted(set([lo,hi]+[E+k*width for k in (-24,-12,-6,-3,-1,-.3,0.,.3,1,3,6,12,24) if lo<E+k*width<hi]))
def signed_difference(a,b):
 return math.exp(a)*(-math.expm1(b-a)) if a>=b else -math.exp(b)*(-math.expm1(a-b))
def events(sp,E,Ep,rad,order,coordinate):
 pair=nc.pair_from_species(sp);xs,ws=nodes(order);lf,lb=rad.logs(E);lp,bp=rad.logs(Ep)
 if coordinate=='q':
  lo,hi=abs(E-Ep),E+Ep;qs=lo+(xs+1)*(hi-lo)/2;cs=(E*E+Ep*Ep-qs*qs)/(2*E*Ep);weights=ws*(hi-lo)/2*qs/(E*Ep)
 else:cs=xs;weights=ws
 result=np.zeros(4)
 for c,w in zip(cs,weights):
  c=float(min(1.,max(-1.,c)))
  forward=math.log(nc.DOS)+2*math.log(E)+nc.log_kernel(pair,E,Ep,c)+lf+bp
  reverse=math.log(nc.DOS)+2*math.log(Ep)+nc.log_kernel(pair,Ep,E,c)+lp+lb
  J=signed_difference(forward,reverse);delta=forward-reverse
  result+=float(w)*np.array([J,math.exp(forward),math.exp(reverse),J*delta])
 return result

def integrate(sp,rad,*,method='paired',cutoff=72.,energy_order=8,inner_order=16,angle_order=32,budget=105.):
 control(cutoff,energy_order,angle_order,inner_order)
 if method not in ('paired','direct'):raise ValueError('paired or direct required')
 start=time.monotonic();source_hashes=hashes();xs,ws=nodes(energy_order);ys,vs=nodes(inner_order)
 # Outputs Nrad,Erad,Srad,Qbath,Stotal,grossforward,grossreverse.
 total=np.zeros(7);outer_panels=[]
 for a,b in zip(energy_edges(cutoff)[:-1],energy_edges(cutoff)[1:]):
  partial=np.zeros(7)
  for x,w in zip(xs,ws):
   E=float(a+(x+1)*(b-a)/2);lf,lb=rad.logs(E);hprime=lb-lf
   lo=E if method=='paired' else 0.;es=partner_edges(sp,E,lo,cutoff)
   for c,d in zip(es[:-1],es[1:]):
    for y,v in zip(ys,vs):
     Ep=float(c+(y+1)*(d-c)/2);J,forward,reverse,hprod=events(sp,E,Ep,rad,angle_order,'q' if method=='paired' else 'cos');lp,bp=rad.logs(Ep)
     if method=='paired':value=np.array([0.,(Ep-E)*J,((bp-lp)-hprime)*J,(E-Ep)*J,hprod,forward,reverse])
     else:value=np.array([-J,-E*J,-hprime*J,E*J,-hprime*J+E*J/sp.T,forward,reverse])
     partial+=float(w*(b-a)/2*v*(d-c)/2)*value
   if time.monotonic()-start>budget:raise TimeoutError('Global exchange experiment budget exhausted')
  total+=partial;outer_panels.append(dict(lo=a,hi=b,values=partial.tolist()))
 if hashes()!=source_hashes:raise RuntimeError('Physical source changed during experiment')
 return dict(method=method,cutoff=cutoff,energy_order=energy_order,inner_order=inner_order,angle_order=angle_order,values=total.tolist(),labels=['Nrad_fm3_s','Erad_MeV_fm3_s','Srad_kB_fm3_s','Qbath_MeV_fm3_s','Stotal_kB_fm3_s','gross_forward_fm3_s','gross_reverse_fm3_s'],outer_panels=outer_panels,runtime_seconds=time.monotonic()-start,source_hashes=source_hashes,bath=sp.asdict(),radiation=rad.metadata())
