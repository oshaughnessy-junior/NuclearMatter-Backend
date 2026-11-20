"""Held-out GM1 review: independently rebuilt delta-shell/spinor integral."""
from __future__ import annotations
from pathlib import Path
import sys,math,json,time,hashlib
import numpy as np
from scipy.special import expit
from scipy.optimize import brentq
from numpy.polynomial.legendre import leggauss
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'gm1_differential_response'));import adapter as A
sys.path.insert(0,str(ROOT/'relativistic_delta_oracle'));import oracle as O
P=Path(__file__).resolve().parent
checks=[]
def check(name,error,tol):checks.append(dict(name=name,error=float(abs(error)),limit=tol,passed=abs(error)<=tol))
def relative(x,y):return np.max(np.abs(x-y))/max(float(np.max(np.abs(y))),1e-300)
def independent(s,Enu,El,c,ml,mul,anti=False,order=24,tail=65):
 # Cylindrical transverse momentum, numerical original energy-delta root.
 M=s.dirac_mass;T=s.T;nu_a=s.total_mu_a-s.potential_a;nu_b=s.total_mu_b-s.potential_b
 pl=math.sqrt(El*El-ml*ml);q=math.sqrt(Enu*Enu+pl*pl-2*Enu*pl*c);omega=Enu-El;w=omega+s.potential_a-s.potential_b
 a=q*q-w*w;assert a>0
 kz=(Enu*Enu+q*q-pl*pl)/(2*q);kx=math.sqrt(Enu*Enu-kz*kz);k=np.array([Enu,kx,0.,kz]);l=np.array([El,kx,0.,kz-q]);qs=np.array([w,0.,0.,q]);t=omega*omega-q*q;Wm=s.weak_bare_mass
 # Independent literal source Eq26-27, no adapter FormFactors/vertices.
 den=(1-t/(4*Wm*Wm))*(1-t/840.**2)**2;gv=(1-4.706*t/(4*Wm*Wm))/den;ga=1.2723/(1-t/1000.**2)**2;f2=3.706/den;gp=2*Wm*Wm*ga/(139.57**2-t)
 V=np.array([gv*O.G[u]-ga*O.G[u]@O.G5-f2/(4*Wm)*sum((O.G[u]@O.G[v]-O.G[v]@O.G[u])*O.MET[v]*qs[v]for v in range(4))-gp/Wm*qs[u]*O.G5 for u in range(4)])
 uk=O.spinors(k,0,anti);ul=O.spinors(l,ml,anti)
 B=np.array([(uk.conj().T@O.G[0]@O.G[u]@(O.I-O.G5)@ul if anti else ul.conj().T@O.G[0]@O.G[u]@(O.I-O.G5)@uk).reshape(-1)for u in range(4)]);L=B@B.conj().T
 # Common-mD spacelike shell has unique longitudinal root for every r.
 z0=-math.sqrt(a)/2;ep0=math.sqrt(M*M+z0*z0);E0=(q*ep0+w*z0)/math.sqrt(a);Emax=max(E0,nu_a,nu_b-w)+tail*T
 def u_at(E):return (((E*math.sqrt(a)-w*z0)/q)**2-M*M-z0*z0)/(2*M)
 cuts=sorted({0.,u_at(Emax)}|{u_at(E)for mu in(nu_a,nu_b-w)for n in(-18,-7,-2,0,2,7,18)if E0<(E:=mu+n*T)<Emax})
 x,wx=leggauss(order);W=np.zeros((2,4,4),complex);maxdelta=0.
 for lo,hi in zip(cuts[:-1],cuts[1:]):
  for xx,ww in zip(x,wx):
   u=lo+(xx+1)*(hi-lo)/2;r=math.sqrt(max(0.,2*M*u))
   def delta(z):return math.sqrt(M*M+r*r+(z+q)**2)-math.sqrt(M*M+r*r+z*z)-w
   bound=1e6;z=brentq(delta,-bound,bound,xtol=1e-9,rtol=1e-14)
   Ea=math.sqrt(M*M+r*r+z*z);Eb=math.sqrt(M*M+r*r+(z+q)**2);jac=abs((z+q)/Eb-z/Ea)
   maxdelta=max(maxdelta,abs((Eb+s.potential_b)-(Ea+s.potential_a)-omega))
   occ=np.array([expit((nu_a-Ea)/T)*expit((Eb-nu_b)/T),expit((nu_b-Eb)/T)*expit((Ea-nu_a)/T)])
   for phi in 2*math.pi*np.arange(8)/8:
    pa=np.array([Ea,r*math.cos(phi),r*math.sin(phi),z]);pb=np.array([Eb,pa[1],pa[2],z+q]);ua=O.spinors(pa,M);ub=O.spinors(pb,M)
    J=np.array([(ub.conj().T@O.G[0]@v@ua).reshape(-1)for v in V]);H=J@J.conj().T
    W+=occ[:,None,None]*H[None,:,:]*M*ww*(hi-lo)/2*(2*math.pi/8)/(16*math.pi**2*Ea*Eb*jac)
 contractions=np.real(np.sum(O.MET[:,None]*O.MET[None,:]*L[None,:,:]*W,axis=(1,2)))
 coeff=O.GF**2*O.VUD**2/(32*math.pi**2)*pl/Enu/O.HBAR*contractions*np.array([expit((El-mul)/T),expit((mul-El)/T)])
 return dict(W=W,L=L,chi=float(coeff[0]),j=float(coeff[1]),max_delta=maxdelta,physical_t=t,kinetic_t=w*w-q*q,formfactors=[gv,ga,f2,gp],order=order,tail=tail)

t0=time.monotonic();matter=A.matter_state(8.,.22,.17,.04);n=matter.nuclear
check('independent_lepton_charge',(matter.electron.net_density+matter.muon.net_density)/n.nB-n.Yp,2e-10)
check('common_bare_mass',n.dirac_mass_mev+n.phi_mev-939.,2e-11)
Un=n.omega_mev-.5*n.rho_mev;Up=n.omega_mev+.5*n.rho_mev;nn=n.nB*(1-n.Yp);np_=n.nB*n.Yp
check('vector_particle_sum_is_twice_field',Un*nn+Up*np_-2*n.vector_field_energy_density,2e-10)
check('nuclear_energy_field_ledger',n.energy_density-n.kinetic_energy_density-n.scalar_field_energy_density-n.vector_field_energy_density,2e-10)
check('nuclear_Euler',n.energy_density+n.pressure-8*n.entropy_density-n.mu_n_mev*nn-n.mu_p_mev*np_,2e-9)
rows=[]
for anti,Enu,El,c in [(False,180.,220.,.1),(True,240.,180.,-.2)]:
 s=matter.ordered(anti);mul=(-1 if anti else 1)*matter.muon.chemical_potential;ml=A.leptons.MUON_MASS
 auth=A.differential(s,Enu,El,c,ml,mul,antineutrino=anti,energy_order=32,phi_order=8,tail=65.)
 refs=[independent(s,Enu,El,c,ml,mul,anti,order=o)for o in(16,28)]
 ref=refs[-1]
 check('independent_original_delta_'+str(anti),ref['max_delta'],2e-8)
 for name in('chi','j'):
  check('independent_coefficient_'+name+'_'+str(anti),auth[name]['value']/ref[name]-1,3e-7)
  check('oracle_refinement_'+name+'_'+str(anti),refs[0][name]/ref[name]-1,3e-7)
 for mode,name in enumerate(('absorption','emission')):check('independent_W_'+name+'_'+str(anti),relative(auth['response'][name]['tensor'],ref['W'][mode]),3e-7)
 check('explicit_spinor_L_'+str(anti),relative(auth['lepton_tensor'],ref['L']),2e-12)
 check('physical_formfactors_'+str(anti),relative(np.array(auth['response']['form_factors']),np.array(ref['formfactors'])),2e-14)
 check('chemical_KMS_total_'+str(anti),math.log(ref['j']/ref['chi'])+(Enu-(mul+s.total_mu_b-s.total_mu_a))/8,2e-11)
 shifted=replace(s,potential_a=s.potential_a+91.3,potential_b=s.potential_b+91.3,total_mu_a=s.total_mu_a+91.3,total_mu_b=s.total_mu_b+91.3)
 sr=A.differential(shifted,Enu,El,c,ml,mul,antineutrino=anti,energy_order=32,phi_order=8,tail=65.)
 check('common_vector_shift_'+str(anti),sr['chi']['value']/auth['chi']['value']-1,2e-12)
 check('bare_weak_mass_'+str(anti),s.weak_bare_mass-939.,0.)
 rows.append(dict(anti=anti,inputs=auth['inputs'],state=vars(s),author_chi=auth['chi']['value'],author_j=auth['j']['value'],independent=refs))
split=A.matter_state(8.,.22,.19,.02)
check('same_Yp_nuclear_mass',split.nuclear.dirac_mass_mev-n.dirac_mass_mev,2e-9)
check('same_Yp_total_nuclear_energy',split.nuclear.energy_density-n.energy_density,2e-9)
assert split.electron.chemical_potential!=matter.electron.chemical_potential and split.muon.chemical_potential!=matter.muon.chemical_potential
for anti in(False,True):
 z=A.OrderedState(9.,939.,0.,0.,947.,932.,939.)
 ar=A.differential(z,175.,150.,-.1,A.leptons.MUON_MASS,25.,antineutrino=anti)
 fr=A.free.differential(A.free.State(9.,939.,939.,947.,932.,939.),175.,150.,-.1,A.leptons.MUON_MASS,25.,antineutrino=anti)
 for name in('j','chi'):check('zero_field_common939_'+name+'_'+str(anti),ar[name]['value']/fr[name]['value']-1,2e-13)
def encode(o):
 if isinstance(o,np.ndarray):return dict(real=o.real.tolist(),imag=o.imag.tolist())if np.iscomplexobj(o)else o.tolist()
 if isinstance(o,np.generic):return o.item()
 raise TypeError(type(o).__name__)
out=dict(checks=checks,passed=sum(c['passed']for c in checks),failed=sum(not c['passed']for c in checks),matter=matter.asdict(),rows=rows,seconds=time.monotonic()-t0)
(P/'results.json').write_text(json.dumps(out,indent=2,default=encode));print(out['passed'],out['failed'],out['seconds'])
if out['failed']:raise SystemExit(1)
