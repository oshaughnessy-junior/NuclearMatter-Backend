"""Controlled free npe binding of two exact CPU modules; no muon gas present."""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass,asdict
import sys,json,hashlib,math,numbers
P=Path(__file__).resolve().parent;sys.dont_write_bytecode=True
LOCK=json.loads((P/'LOCK.json').read_text());PKG=P/'package_copy';sys.path.insert(0,str(PKG/'src'))
for rel,h in json.loads((P/'PACKAGE_PIN.json').read_text())['all_python'].items():
 if hashlib.sha256((PKG/rel).read_bytes()).hexdigest()!=h:raise RuntimeError('Owned package source mismatch: '+rel)
from nmbackend.reference import finite_t_leptons as L,finite_t_baryons as B
from nmbackend import constants as C
if not all(str(m.__file__).startswith(str(PKG/'src'))for m in[L,B,C]):raise RuntimeError('Actual package origin mismatch')
MN,MP,ME,HC=[LOCK['constants'][x]for x in ['mn','mp','me','HBARC']]
if C.HBARC!=HC:raise RuntimeError('Unit convention mismatch')
@dataclass(frozen=True)
class Matter:
 T:float;nB:float;Ye:float;photons:bool
 neutron:B.ParticleState;proton:B.ParticleState;electron:L.LeptonState
 energy:float;pressure:float;entropy:float;cv:float
 e_Y:float;e_n:float;h_n:float;h_p:float;h_e:float
 photon_energy:float;photon_pressure:float;photon_entropy:float;photon_cv:float
 def asdict(self):return asdict(self)
 @property
 def entropy_per_baryon(self):return self.entropy/self.nB
 @property
 def specific_energy(self):return self.energy/self.nB
 def weak_bath(self):
  return dict(T=self.T,mu_n=self.neutron.chemical_potential,mu_p=self.proton.chemical_potential,mu_e=self.electron.chemical_potential)
def state(T,nB,Ye,*,photons=True):
 if not all(isinstance(x,numbers.Real)and not isinstance(x,bool)and math.isfinite(x)for x in[T,nB,Ye])or type(photons)is not bool:raise ValueError('Finite real state and literal photon closure required')
 d=LOCK['admission']
 if not d['T'][0]<=T<=d['T'][1]or not d['nB'][0]<=nB<=d['nB'][1]or not d['Ye_open'][0]<Ye<d['Ye_open'][1]:raise ValueError('Outside selected research admission')
 control=LOCK['EOS_controls'];n=B.state_density(T,(1-Ye)*nB,mass_mev=MN,**control);p=B.state_density(T,Ye*nB,mass_mev=MP,**control);e=L.state_density(T,Ye*nB,mass_mev=ME,**control)
 h_n=n.chemical_potential-T*n.dmu_dtemperature_fixed_particle_density;h_p=p.chemical_potential-T*p.dmu_dtemperature_fixed_particle_density;h_e=e.chemical_potential-T*e.dmu_dtemperature_fixed_net_density
 eg=math.pi**2*T**4/(15*HC**3)if photons else 0.;pg=eg/3;sg=4*eg/(3*T);cg=4*eg/T
 energy=n.energy_density+p.energy_density+e.energy_density+eg;pressure=n.pressure+p.pressure+e.pressure+pg;entropy=n.entropy_density+p.entropy_density+e.entropy_density+sg;cv=n.cv_fixed_particle_density+p.cv_fixed_particle_density+e.cv_fixed_net_density+cg
 return Matter(float(T),float(nB),float(Ye),photons,n,p,e,energy,pressure,entropy,cv,nB*(h_p-h_n+h_e),(1-Ye)*h_n+Ye*(h_p+h_e),h_n,h_p,h_e,eg,pg,sg,cg)
def compression_rhs(t,T,*,n0=1e-7,Ye=.3,photons=True,tau=1.):
 n=n0*math.exp(math.log(2)*math.sin(2*math.pi*t/tau));ndot=n*math.log(2)*2*math.pi/tau*math.cos(2*math.pi*t/tau);s=state(T,n,Ye,photons=photons)
 # d(e/n)/dt=-P*d(1/n)/dt, no weak/radiative sources.
 return ((s.energy+s.pressure)/n-s.e_n)*ndot/s.cv,s,ndot
