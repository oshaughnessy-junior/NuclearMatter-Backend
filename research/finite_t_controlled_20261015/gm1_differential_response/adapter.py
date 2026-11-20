"""Selected GM1 Hartree differential CC response, explicit Guo current ansatz."""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass,asdict
import math,sys
import numpy as np
from scipy.special import log_expit
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True
for directory in ('rmf_reference','rmf_phase','thermodynamics','relativistic_tensor'):sys.path.insert(0,str(ROOT/directory))
import gm1_finite_t as eos
import stability
import lepton_reference as leptons
import tensor as free
BARE_MASS=939.
@dataclass(frozen=True)
class OrderedState:
 T:float
 dirac_mass:float
 potential_a:float
 potential_b:float
 total_mu_a:float
 total_mu_b:float
 weak_bare_mass:float
 def __post_init__(self):
  if not all(math.isfinite(x)for x in asdict(self).values())or min(self.T,self.dirac_mass,self.weak_bare_mass)<=0:raise ValueError('Finite quasiparticle parameters andpositive T,mD,Mweak required')

@dataclass(frozen=True)
class Matter:
 nuclear:eos.NuclearState
 electron:leptons.State
 muon:leptons.State
 Ye:float
 Ymu:float
 nuclear_hessian:tuple
 stability_diagnostics:dict
 forward_gates:dict
 def ordered(self,antineutrino=False):
  if not isinstance(antineutrino,bool):raise ValueError('Explicit antineutrino bool required')
  n=self.nuclear;Un=n.omega_mev-.5*n.rho_mev;Up=n.omega_mev+.5*n.rho_mev
  return OrderedState(n.temperature,n.dirac_mass_mev,Up if antineutrino else Un,Un if antineutrino else Up,n.mu_p_mev if antineutrino else n.mu_n_mev,n.mu_n_mev if antineutrino else n.mu_p_mev,BARE_MASS)
 def asdict(self):return asdict(self)
 @property
 def energy_density(self):return self.nuclear.energy_density+self.electron.energy_density+self.muon.energy_density
 @property
 def pressure(self):return self.nuclear.pressure+self.electron.pressure+self.muon.pressure
 @property
 def entropy_density(self):return self.nuclear.entropy_density+self.electron.entropy_density+self.muon.entropy_density

def matter_state(T,nB,Ye,Ymu):
 if not all(math.isfinite(x)for x in(T,nB,Ye,Ymu))or T<=0 or nB<=0 or not 0<Ye+Ymu<1:raise ValueError('FiniteT,nB>0 and0<Ye+Ymu<1 required')
 n=eos.state(nB,T,Ye+Ymu);H,detail=stability.implicit_hessian(n);eigen=np.linalg.eigvalsh(H);D=detail['canonical_scalar_curvature_fm_minus2']
 if eigen[0]<=0 or D<=0:raise ValueError('Selected differential interface requires locally stable nuclear branch')
 gn=eos.gas(T/eos.HBARC,n.kinetic_mu_n_mev/eos.HBARC,n.dirac_mass_mev/eos.HBARC);gp=eos.gas(T/eos.HBARC,n.kinetic_mu_p_mev/eos.HBARC,n.dirac_mass_mev/eos.HBARC)
 residual=max(abs(gn['n']/(nB*(1-n.Yp))-1),abs(gp['n']/(nB*n.Yp)-1),n.max_normalized_equation_residual)
 energy=sum((n.kinetic_energy_density,n.scalar_field_energy_density,n.vector_field_energy_density));euler=n.energy_density+n.pressure-T*n.entropy_density-n.mu_n_mev*nB*(1-n.Yp)-n.mu_p_mev*nB*n.Yp
 if residual>2e-9 or abs(energy/n.energy_density-1)>1e-12 or abs(euler)>2e-8 or n.antiparticle_density/nB>1e-8:raise ArithmeticError('Nuclear forward/stability/antiparticle gate failed')
 e=leptons.state_density(T,nB*Ye,leptons.ELECTRON_MASS);mu=leptons.state_density(T,nB*Ymu,leptons.MUON_MASS)
 charge=(e.net_density+mu.net_density)/nB-(Ye+Ymu)
 if abs(charge)>2e-10:raise ArithmeticError('Independent lepton charge-root gate failed')
 return Matter(n,e,mu,Ye,Ymu,tuple(tuple(float(v)for v in row)for row in H),dict(eigenvalues_MeV_fm3=eigen.tolist(),canonical_scalar_curvature_fm_minus2=D,scope='Local nuclear stability, no global coexistence or external GM1 validation'),dict(normalized_nuclear_root_residual=residual,nuclear_Euler_residual_MeV_fm3=euler,field_energy_decomposition_residual=energy-n.energy_density,charge_fraction_residual=charge,antinucleon_ratio=n.antiparticle_density/nB))

class PhysicalFormFactors:
 """Evaluate form factors at physicalq² while the frozen vertex sees kineticq*."""
 def __init__(self,physical_t,model):self.physical_t=physical_t;self.model=model
 def values(self,kinetic_t,Mweak):return self.model.values(self.physical_t,Mweak)

def response(state,omega,q,*,model=free.C.FormFactors(),**controls):
 if not isinstance(state,OrderedState)or not all(math.isfinite(x)for x in(omega,q))or q<=0:raise ValueError('OrderedState andfinite physicalomega,q>0 required')
 w=omega-(state.potential_b-state.potential_a)
 kinetic=free.State(state.T,state.dirac_mass,state.dirac_mass,state.total_mu_a-state.potential_a,state.total_mu_b-state.potential_b,state.weak_bare_mass)
 r=free.response(kinetic,w,q,model=PhysicalFormFactors(omega*omega-q*q,model),**controls)
 r['mapping']=dict(ordered_total_state=asdict(state),physical_omega=omega,kinetic_omega=w,q=q,physical_q2=omega*omega-q*q,kinetic_qstar2=w*w-q*q,prescription='Guo: physicalq² form factors; kineticq* WM/PSvertices; bareMweak independent ofmD',physical_current_scope='Phenomenological Hartree free-quasiparticle ansatz, noRPA/vertexconservation reconstruction')
 return r

def differential(state,Enu,El,cosine,lepton_mass,mu_lepton,*,antineutrino=False,model=free.C.FormFactors(),**controls):
 if not isinstance(antineutrino,bool)or not all(math.isfinite(x)for x in(Enu,El,cosine,lepton_mass,mu_lepton))or Enu<=0 or lepton_mass<0 or El<=lepton_mass or not -1<=cosine<=1:raise ValueError('Finite physicalleptoninputs,El>ml andexplicit anti bool required')
 pl=math.sqrt((El-lepton_mass)*(El+lepton_mass));omega=Enu-El;q=math.sqrt((Enu-pl)**2+2*Enu*pl*(1-cosine));r=response(state,omega,q,model=model,**controls)
 kz=(Enu*Enu+q*q-pl*pl)/(2*q);kx2=Enu*Enu-kz*kz
 if kx2 < -1e-10*Enu*Enu:raise ArithmeticError('Physical leptonqframe is offshell')
 kx=math.sqrt(max(0.,kx2));k=np.array([Enu,kx,0.,kz]);l=np.array([El,kx,0.,kz-q]);V=free.V;left=[g@(V.I-V.G5)for g in V.G];spin=V.slash(l)+(-1 if antineutrino else 1)*lepton_mass*V.I
 L=np.array([[np.trace(spin@left[u]@V.slash(k)@V.adjoint(left[v]))for v in range(4)]for u in range(4)])
 if antineutrino:L=L.T
 pref=V.GF**2*V.VUD**2/(32*math.pi**2)*pl/Enu/free.HBAR
 chi=free._coefficient(r['absorption'],L,log_expit((El-mu_lepton)/state.T),pref);j=free._coefficient(r['emission'],L,log_expit((mu_lepton-El)/state.T),pref)
 return dict(chi=chi,j=j,response=r,lepton_tensor=L,lepton_k=k,lepton_l=l,inputs=dict(Enu=Enu,El=El,cosine=cosine,lepton_mass=lepton_mass,mu_lepton=mu_lepton,antineutrino=antineutrino),units='jchi:s^-1MeV^-1percos;W:MeV²')
