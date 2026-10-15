"""Selected Hartree differential mapping, refinement and thermodynamic gates."""
from adapter import *
from dataclasses import replace
import json,time,hashlib,atexit
P=Path(__file__).resolve().parent;start=time.monotonic();checks=[];rows=[];matters=[]
sourcepaths=[Path(__file__),P/'adapter.py',ROOT/'relativistic_tensor/tensor.py',ROOT/'muon_full_current/current.py',ROOT/'muon_vacuum/vacuum_cc.py',ROOT/'rmf_reference/gm1_finite_t.py',ROOT/'rmf_phase/stability.py',ROOT/'thermodynamics/lepton_reference.py'];hashes=lambda:{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in sourcepaths};before=hashes()
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and error<tol)))
def serialize(x):
 if isinstance(x,np.ndarray):return dict(real=x.real.tolist(),imag=x.imag.tolist())if np.iscomplexobj(x)else x.tolist()
 if isinstance(x,np.generic):return x.item()
 raise TypeError(type(x).__name__)
def save():(P/'results.json').write_text(json.dumps(dict(checks=checks,rows=rows,matters=matters,source_hashes_before=before,source_hashes_after=hashes(),runtime_seconds=time.monotonic()-start),indent=2,default=serialize)+'\n')
atexit.register(save)
for T,n,Ye,Ymu in [(5.,.2,.18,.02),(12.,.24,.20,.05)]:
 m=matter_state(T,n,Ye,Ymu);matters.append(m.asdict());nu=m.nuclear
 gate('EOS root gate'+str(T),m.forward_gates['normalized_nuclear_root_residual'],2e-9)
 gate('nuclear local Hessian positive'+str(T),float(m.stability_diagnostics['eigenvalues_MeV_fm3'][0]<=0),.1)
 gate('canonical scalar curvature positive'+str(T),float(m.stability_diagnostics['canonical_scalar_curvature_fm_minus2']<=0),.1)
 gate('functional Euler'+str(T),abs(m.forward_gates['nuclear_Euler_residual_MeV_fm3']),2e-8)
 gate('independent charge roots'+str(T),abs(m.forward_gates['charge_fraction_residual']),2e-10)
 Un=nu.omega_mev-.5*nu.rho_mev;Up=nu.omega_mev+.5*nu.rho_mev;vector_sum=Un*n*(1-nu.Yp)+Up*n*nu.Yp
 gate('vector occupation sum is2fieldenergy'+str(T),abs(vector_sum/(2*nu.vector_field_energy_density)-1),3e-12)
 gate('correct quasiparticle energy correction'+str(T),abs((nu.kinetic_energy_density+vector_sum+nu.scalar_field_energy_density-nu.vector_field_energy_density)/nu.energy_density-1),3e-12)
 points=[('electron',110.,160.,.3,m.electron,False),('muon',160.,210.,-.25,m.muon,False)]if T==5 else[('antimuon',200.,148.,.2,m.muon,True)]
 for label,E,El,c,lepton,anti in points:
  state=m.ordered(anti);mul=-lepton.chemical_potential if anti else lepton.chemical_potential;row=dict(label=label,state=asdict(state),matter_index=len(matters)-1,runs=[]);rows.append(row)
  for order,phi,tail in [(24,8,50.),(40,16,80.)]:
   r=differential(state,E,El,c,lepton.mass,mul,antineutrino=anti,energy_order=order,phi_order=phi,tail=tail);row['runs'].append(r)
   gate(label+' separately integrated KMS'+str(order),abs(r['j']['log']-r['chi']['log']+(E-(mul+state.total_mu_b-state.total_mu_a))/state.T),3e-10)
   for branch in ('absorption','emission'):
    W=r['response'][branch]['tensor'];scale=np.max(abs(W));gate(label+branch+' Hermitian'+str(order),np.max(abs(W-W.conj().T))/scale,3e-12);gate(label+branch+' PSD'+str(order),max(0.,-np.linalg.eigvalsh((W+W.conj().T)/2)[0])/scale,3e-11)
  a,b=row['runs'];
  for key in ('j','chi'):gate(label+' coefficient refinement '+key,abs(math.expm1(a[key]['log']-b[key]['log'])),3e-7)
  for branch in ('absorption','emission'):
   X=a['response'][branch]['tensor'];Y=b['response'][branch]['tensor'];gate(label+branch+' all16 refinement',np.max(abs(X-Y))/np.max(abs(Y)),3e-7)
  # Constant common vector/total-mu shift leaves all microscopic occupations intact.
  shifted=replace(state,potential_a=state.potential_a+37.,potential_b=state.potential_b+37.,total_mu_a=state.total_mu_a+37.,total_mu_b=state.total_mu_b+37.)
  rr=differential(shifted,E,El,c,lepton.mass,mul,antineutrino=anti,energy_order=40,phi_order=16,tail=80.);row['common_shift_result']=rr
  for key in ('j','chi'):gate(label+' common vector shift '+key,abs(rr[key]['log']-b[key]['log']),3e-11)
  for branch in ('absorption','emission'):gate(label+' common vector all16 '+branch,np.max(abs(rr['response'][branch]['tensor']-b['response'][branch]['tensor']))/np.max(abs(b['response'][branch]['tensor'])),3e-11)
  mapping=b['response']['mapping'];gate(label+' exact kinetic energy conservation',abs(mapping['kinetic_omega']+(state.potential_b-state.potential_a)-(E-El)),3e-12)
  gate(label+' physicalq2 differsfromkineticqstar2',float(abs(mapping['physical_q2']-mapping['kinetic_qstar2'])<1.),.1)
  gate(label+' bare weak mass939',abs(state.weak_bare_mass-939.),1e-12);gate(label+' weak mass distinctfromDirac',float(abs(state.weak_bare_mass-state.dirac_mass)<100.),.1)
  # Deliberately different model choices, not uncertainty estimates or fixes.
  wrongmass=replace(state,weak_bare_mass=state.dirac_mass);wr=differential(wrongmass,E,El,c,lepton.mass,mul,antineutrino=anti,energy_order=40,phi_order=16,tail=80.)
  kinetic=free.State(state.T,state.dirac_mass,state.dirac_mass,state.total_mu_a-state.potential_a,state.total_mu_b-state.potential_b,state.weak_bare_mass);wrongff=free.response(kinetic,mapping['kinetic_omega'],mapping['q'],energy_order=40,phi_order=16,tail=80.)
  diffmass=abs(math.expm1(wr['chi']['log']-b['chi']['log']));diffff=np.max(abs(wrongff['absorption']['tensor']-b['response']['absorption']['tensor']))/np.max(abs(b['response']['absorption']['tensor']));row['deliberate_different_models']=dict(weak_mass_equals_Dirac_chi_relative_difference=diffmass,kineticq2_formfactor_tensor_difference=diffff)
  gate(label+' no silent weak mass substitution witness',float(diffmass<1e-6),.1);gate(label+' no silent formfactorq substitution witness',float(diffff<1e-6),.1)
  save()
# Exactly matched zero-field common-bare-mass free limit; no physicalQ channel.
for anti in (False,True):
 state=OrderedState(8.,939.,0.,0.,946.,941.,939.);args=(145.,121.,-.2,leptons.MUON_MASS,-23. if anti else 75.)
 a=differential(state,*args,antineutrino=anti,energy_order=40,phi_order=16,tail=80.);b=free.differential(free.State(8.,939.,939.,946.,941.,939.),*args,antineutrino=anti,energy_order=40,phi_order=16,tail=80.)
 for key in ('j','chi'):gate('zero field matched free '+str(anti)+key,abs(a[key]['log']-b[key]['log']),3e-12)
 for branch in ('absorption','emission'):gate('zero field matched allW '+str(anti)+branch,np.max(abs(a['response'][branch]['tensor']-b['response'][branch]['tensor']))/np.max(abs(b['response'][branch]['tensor'])),3e-12)
 gate('zero field allL '+str(anti),np.max(abs(a['lepton_tensor']-b['lepton_tensor']))/np.max(abs(b['lepton_tensor'])),3e-12)
gate('all physical sourcehashes unchanged',float(hashes()!=before),.1);save();print(sum(c['passed']for c in checks),'/',len(checks),'runtime',time.monotonic()-start)
for c in checks:
 if not c['passed']:print('FAIL',c)
