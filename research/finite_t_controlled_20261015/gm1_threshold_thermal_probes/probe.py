"""Actual finite-T differential response around reviewed cold GM1 onset.

Fixed cold fractions, two explicit field closures, exact cold occupation cuts.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import asdict
import sys,math,json,time,hashlib,platform
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];P=Path(__file__).resolve().parent;sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'gm1_beta_thresholds'));import cold_beta as B
sys.path.insert(0,str(ROOT/'gm1_differential_response'));import adapter as A
sys.path.insert(0,str(ROOT/'relativistic_free_eos'));import free_baryons as F
checks=[]
def ck(name,error,tol):checks.append(dict(name=name,error=abs(float(error)),limit=tol,passed=bool(abs(error)<=tol)))

def kinematics(cold,Enu=.3,cosine=-1.):
 El=cold.mu_lepton;ml=B.ME;pl=math.sqrt((El-ml)*(El+ml));q=math.sqrt((Enu-pl)**2+2*Enu*pl*(1-cosine))
 return dict(Enu=Enu,El=El,cosine=cosine,lepton_mass=ml,q=q,omega=Enu-El)

def cold_window(cold,kin,*,mass=None,Ua=None,Ub=None):
 n=cold.nuclear;M=n.dirac_mass_mev if mass is None else mass
 Ua=n.omega_mev-.5*n.rho_mev if Ua is None else Ua;Ub=n.omega_mev+.5*n.rho_mev if Ub is None else Ub
 q=kin['q'];w=kin['omega']+Ua-Ub
 kn=B.H*B.gm.kf(cold.nB*(1-cold.Yp));kp=B.H*B.gm.kf(cold.nB*cold.Yp)
 nua=math.hypot(M,kn);nub=math.hypot(M,kp)
 if q<=abs(w):return dict(classification='empty_particle_hole_support',q=q,w=w)
 aa=(q-abs(w))*(q+abs(w));Emin=-w/2+q/2*math.sqrt(1+4*M*M/aa)
 lo=max(Emin,nub-w);hi=nua;elo=max(Emin,nua);ehi=nub-w
 # Independent scalar root of original shell cosine, no free.support call.
 def shell_boundary(E):
  p=math.sqrt((E-M)*(E+M));return abs((2*w*E+w*w-q*q)/(2*q*p))-1
 numerical=brentq(shell_boundary,M*(1+1e-13),M+1e6,xtol=5e-11,rtol=2e-14)
 ck('independent_cold_support_Emin',numerical-Emin,3e-8)
 witness=None
 if hi>lo:
  E=(lo+hi)/2;p=math.sqrt((E-M)*(E+M));c=(2*w*E+w*w-q*q)/(2*q*p);Eb=math.sqrt(M*M+p*p+q*q+2*p*q*c)
  ck('original_total_energy_cold_witness',(Eb+Ub)-(E+Ua)-kin['omega'],3e-9)
  assert abs(c)<1 and E<nua and Eb>nub
  witness=dict(Ea_star=E,Eb_star=Eb,cosine_pq=c)
 return dict(classification='cold_absorption_window_open'if hi>lo else'cold_absorption_window_EMPTY',q=q,w=w,dirac_mass=M,potential_a=Ua,potential_b=Ub,nu_a_Fermi=nua,nu_b_Fermi=nub,Emin=Emin,Emin_independent_root=numerical,absorption_lower=lo,absorption_upper=hi,absorption_signed_gap=hi-lo,absorption_width=max(0.,hi-lo),emission_lower=elo,emission_upper=ehi,emission_width=max(0.,ehi-elo),witness=witness,lepton_edge='El equals cold electron chemical potential; T0 edge value convention1/2, irrelevant to integrated phase volume')

def direct_density(T,mu,m,pairs=False):
 Emax=max(m,abs(mu))+70*T;pmax=math.sqrt((Emax-m)*(Emax+m))
 return quad(lambda p:p*p*(expit((mu-math.hypot(p,m))/T)-(expit((-mu-math.hypot(p,m))/T) if pairs else 0.)),0,pmax,epsabs=1e-6,epsrel=3e-11,limit=300)[0]/(math.pi**2*B.H**3)

def states(cold,T):
 n=cold.nuclear;M=n.dirac_mass_mev;Un=n.omega_mev-.5*n.rho_mev;Up=n.omega_mev+.5*n.rho_mev
 neutron=F.state_density(T,cold.nB*(1-cold.Yp),M);proton=F.state_density(T,cold.nB*cold.Yp,M)
 e=A.leptons.state_density(T,cold.nB*cold.Ye,B.ME);mu=A.leptons.state_density(T,cold.nB*cold.Ymu,B.MMU)
 frozen=A.OrderedState(T,M,Un,Up,neutron.chemical_potential+Un,proton.chemical_potential+Up,939.)
 warm=A.matter_state(T,cold.nB,cold.Ye,cold.Ymu)
 return[dict(closure='frozen_COLD_fields_densitymatched_thermal_populations',state=frozen,electron=e,muon=mu,metadata=dict(neutron=neutron.asdict(),proton=proton.asdict(),warning='Prescribed cold Hamiltonian, not a self-consistent finite-T Hartree EOS')),dict(closure='selfconsistent_warm_GM1_fixed_cold_fractions',state=warm.ordered(),electron=warm.electron,muon=warm.muon,metadata=warm.asdict())]

def save_json(path,value):
 def encode(o):
  if isinstance(o,np.ndarray):return dict(real=o.real.tolist(),imag=o.imag.tolist())if np.iscomplexobj(o)else o.tolist()
  if isinstance(o,np.generic):return o.item()
  if hasattr(o,'asdict'):return o.asdict()
  raise TypeError(type(o).__name__)
 path.write_text(json.dumps(value,indent=2,default=encode))

def main():
 start=time.monotonic();critical=B.threshold('e',.26,.30);rows=[];coldrows=[]
 for factor in(.97,1.,1.03):
  cold=B.state(factor*critical.nB);kin=kinematics(cold);window=cold_window(cold,kin);coldrows.append(dict(density_factor=factor,cold=cold.asdict(),kinematics=kin,cold_window=window))
  assert window['emission_width']==0
  if factor==.97:assert window['absorption_width']==0
  for T in(.1,.3,1.):
   for item in states(cold,T):
    s=item['state'];e=item['electron'];mu=item['muon'];controls=[]
    for order,tail in[(24,50.),(40,80.)]:controls.append(A.differential(s,kin['Enu'],kin['El'],kin['cosine'],kin['lepton_mass'],e.chemical_potential,energy_order=order,phi_order=4,tail=tail))
    instant=cold_window(cold,kin,mass=s.dirac_mass,Ua=s.potential_a,Ub=s.potential_b)
    for name in('j','chi'):
     assert all(r[name]['log']is not None and math.isfinite(r[name]['log'])for r in controls)
     ck('independent_log_rate_refinement_'+name,controls[0][name]['log']-controls[1][name]['log'],3e-7)
    for r in controls:
     affinity=e.chemical_potential+s.total_mu_b-s.total_mu_a
     ck('integrated_independent_occupations_KMS',r['j']['log']-r['chi']['log']+(kin['Enu']-affinity)/T,3e-10)
    targets=[cold.nB*(1-cold.Yp),cold.nB*cold.Yp]
    for chem,target in zip([s.total_mu_a-s.potential_a,s.total_mu_b-s.potential_b],targets):ck('independent_particle_density',direct_density(T,chem,s.dirac_mass)/target-1,3e-9)
    for lep,target,mass in[(e,cold.nB*cold.Ye,B.ME),(mu,cold.nB*cold.Ymu,B.MMU)]:ck('independent_pair_net_density',direct_density(T,lep.chemical_potential,mass,True)/target-1,3e-9)
    row=dict(density_factor=factor,nB=cold.nB,T=T,closure=item['closure'],state=asdict(s),electron=e.asdict(),muon=mu.asdict(),metadata=item['metadata'],cold_reference_window=window,instantaneous_fields_T0_density_window=instant,kinematics=kin,affinities=dict(e=e.chemical_potential+s.total_mu_b-s.total_mu_a,mu=mu.chemical_potential+s.total_mu_b-s.total_mu_a),runs=controls,thermal_opening_of_cold_reference=window['absorption_width']==0,thermal_opening_at_same_instantaneous_Hamiltonian=instant['absorption_width']==0)
    rows.append(row);save_json(P/'partial_results.json',dict(cold_onset=critical.asdict(),cold_states=coldrows,rows=rows,checks=checks));print(factor,T,item['closure'],controls[-1]['chi']['log'],instant['absorption_signed_gap'],flush=True)
 # Same finite incident-energy/selected-angle cold opening, separate from DU triangle.
 def finite_gap(n):
  c=B.state(n);return cold_window(c,kinematics(c))['absorption_signed_gap']
 onset=brentq(finite_gap,.97*critical.nB,critical.nB,xtol=5e-13,rtol=2e-14)
 out=dict(cold_onset=critical.asdict(),finite_incident_selected_probe_cold_onset=onset,cold_states=coldrows,rows=rows,checks=checks,passed=sum(c['passed']for c in checks),failed=sum(not c['passed']for c in checks),seconds=time.monotonic()-start,scope='Selected differential nu_e capture/inverse, fixed cold fractions; not warm beta equilibrium, full Urca rate or width')
 save_json(P/'results.json',out);print(out['passed'],out['failed'],out['seconds'])
 if out['failed']:raise SystemExit(1)
if __name__=='__main__':main()
