"""Static unregulated NR OPE Born operator; no weak current/rate.

Spin tensor product precedes isospin tensor product, each ordered (1,2).
Nucleon isospin basis (n,p), tau3=diag(+1,-1). Momenta MeV, T MeV^-2.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
PAULI=np.array([[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]],complex)
I2=np.eye(2,dtype=complex)
SWAP=np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]],complex)
TAUDOT=sum(np.kron(s,s)for s in PAULI)
EXCHANGE=np.kron(SWAP,SWAP)
ISO={'n':0,'p':1}
@dataclass(frozen=True)
class Model:
 f_piNN:float=1.01
 m_pi:float=140.
 common_mass:float=939.
 def __post_init__(self):
  if not all(math.isfinite(x)for x in(self.f_piNN,self.m_pi,self.common_mass))or self.f_piNN<0 or self.m_pi<=0 or self.common_mass<=0:raise ValueError('Finite nonnegative f and positive pion/common masses required')
 @property
 def coefficient(self):return self.f_piNN**2/self.m_pi**2
 @property
 def pseudoscalar_g(self):return 2*self.common_mass*self.f_piNN/self.m_pi

def vector(q):
 q=np.asarray(q,dtype=float)
 if q.shape!=(3,)or not np.all(np.isfinite(q)):raise ValueError('Finite three-momentum required')
 return q

def spin(q):
 q=vector(q);a=np.einsum('i,ijk->jk',q,PAULI);return np.kron(a,a)

def direct(q,model=Model()):
 q=vector(q);return -model.coefficient*np.kron(spin(q),TAUDOT)/(q@q+model.m_pi**2)

def antisymmetrized(q,k,model=Model()):
 """Final-order labelled amplitude V(q)-P_spin P_iso V(k); not divided by sqrt2.

q=p3-p1, k=p4-p1; exact total momentum conservation is caller's duty.
No energy shell or relativistic kinematic claim is implied by arbitrary q,k.
 """
 return direct(q,model)-EXCHANGE@direct(k,model)

def channel(initial,final,q,k,model=Model(),antisym=True):
 if len(initial)!=2 or len(final)!=2 or any(x not in ISO for x in initial+final):raise ValueError('Explicit two-nucleon n/p channel required')
 i=2*ISO[initial[0]]+ISO[initial[1]];f=2*ISO[final[0]]+ISO[final[1]]
 a=antisymmetrized(q,k,model)if antisym else direct(q,model)
 # Full matrix index spin*4+isospin.
 return a[np.ix_(np.arange(4)*4+f,np.arange(4)*4+i)]

def spin_average(a):
 a=np.asarray(a,dtype=complex)
 if a.shape!=(4,4):raise ValueError('Four-spin by four-spin channel required')
 return float(np.sum(abs(a)**2)/4)

def born_dsigma(initial,final,p,cosine,model=Model()):
 """LO common-mass elastic CM unpolarized dσ/dΩ in fm²/sr.

np->np uses definite outgoing neutron direction over full sphere. nn/pp
use amplitude differential before final identical-particle 1/2 integration.
Not a phenomenological NN fit, exact relativistic cross section or MU rate.
 """
 if not math.isfinite(p)or p<0 or not math.isfinite(cosine)or not -1<=cosine<=1:raise ValueError('Finite p>=0 and cosine in[-1,1]')
 pi=np.array([0.,0.,p]);pf=p*np.array([math.sqrt(1-cosine*cosine),0.,cosine]);q=pf-pi;k=-pf-pi
 t=channel(initial,final,q,k,model)
 return (model.common_mass/2)**2/(4*math.pi**2)*spin_average(t)*197.3269804**2
