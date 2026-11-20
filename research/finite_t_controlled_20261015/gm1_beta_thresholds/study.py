"""Cold composition map, separate threshold brackets and warm state prep."""
from cold_beta import *
import json,time,hashlib,atexit
P=Path(__file__).resolve().parent;start=time.monotonic();checks=[];rows=[];onsets={};warm=[]
paths=[P/'cold_beta.py',Path(__file__),ROOT/'rmf_reference/gm1_finite_t.py',ROOT/'thermodynamics/lepton_reference.py',ROOT/'rmf_reference/source_snapshot/rmft.py',ROOT/'rmf_reference/source_snapshot/constants.py',P/'source_snapshot/leptons.py',ROOT/'gm1_differential_response/adapter.py',ROOT/'rmf_phase/stability.py',ROOT/'relativistic_tensor/tensor.py',ROOT/'muon_full_current/current.py',ROOT/'muon_vacuum/vacuum_cc.py'];hashes=lambda:{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths};before=hashes()
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and error<tol)))
def save():(P/'results.json').write_text(json.dumps(dict(checks=checks,rows=rows,onsets=onsets,warm=warm,source_hashes_before=before,source_hashes_after=hashes(),runtime_seconds=time.monotonic()-start),indent=2)+'\n')
atexit.register(save)
for nB in np.linspace(.08,.8,37):
 s=state(float(nB));rows.append(s.asdict());gate('density'+str(nB)+' neutrality',abs(s.neutrality_residual),2e-11);gate('density'+str(nB)+' beta',s.beta_residual_MeV,1e-10);gate('density'+str(nB)+' nuclear stable',float(s.nuclear_eigenvalues[0]<=0),.1);gate('density'+str(nB)+' charge-relaxed stable',float(s.charge_relaxed_eigenvalues[0]<=0),.1)
 # Cold beta+neutral total energy Euler reduces to mu_n nB.
 gate('density'+str(nB)+' full betaEuler',abs((s.energy_density+s.pressure)/(s.nuclear.mu_n_mev*nB)-1),2e-11)
 for l in (s.electron,s.muon):
  measured=quad(lambda p:p*p,0,l['kF_MeV'],epsabs=1e-20,epsrel=2e-12)[0]/(math.pi**2*H**3)
  gate('density'+str(nB)+' independent momentum leptonmass'+str(l['mass']),abs(measured-l['density'])/nB,2e-12)
 save()
for flavor,lo,hi in [('e',.26,.30),('mu',.32,.38)]:
 s=threshold(flavor,lo,hi);onsets[flavor]=dict(state=s.asdict(),sign_bracket=[getattr(state(x),'delta_'+flavor+'_MeV')for x in(lo,hi)],density_bracket=[lo,hi]);gate(flavor+' actual triangle root',abs(getattr(s,'delta_'+flavor+'_MeV')),3e-8)
 x=(s.Ye if flavor=='e'else s.Ymu)/s.Yp;analyticcrit=1/(1+(1+x**(1/3))**3);gate(flavor+' independent fraction geometry',abs(s.Yp-analyticcrit),3e-11)
 for eps in (.001,.0005):
  left=state(s.nB*(1-eps));right=state(s.nB*(1+eps));gate(flavor+' onset actual signs '+str(eps),float(not(getattr(left,'delta_'+flavor+'_MeV')>0>getattr(right,'delta_'+flavor+'_MeV'))),.1)
 # Independently reconstruct nuclear chemical Hessian by cold composition solves.
 for hfrac in (1e-3,5e-4):
  nn=s.nB*(1-s.Yp);np_=s.nB*s.Yp;Hfd=np.zeros((2,2))
  for j,nj in enumerate((nn,np_)):
   left=np.array([nn,np_]);right=left.copy();left[j]-=hfrac*nj;right[j]+=hfrac*nj
   a=gm.cold_state(sum(left),left[1]/sum(left));b=gm.cold_state(sum(right),right[1]/sum(right));Hfd[:,j]=(np.array([b.mu_n_mev,b.mu_p_mev])-np.array([a.mu_n_mev,a.mu_p_mev]))/(2*hfrac*nj)
  gate(flavor+' independent coldH FD '+str(hfrac),np.max(abs(Hfd-np.array(s.nuclear_hessian)))/np.max(abs(np.asarray(s.nuclear_hessian))),3e-6)
 save()
# Muon appearance is a distinct threshold, not a muon-Urca triangle onset.
root=brentq(lambda n:state(n).mu_lepton-MMU,.12,.16,xtol=1e-13,rtol=1e-13);s=state(root);onsets['muon_appearance']=dict(state=s.asdict(),chemical_threshold_MeV=MMU,chemical_residual_MeV=s.mu_lepton-MMU)
gate('muon appearance chemical root',abs(s.mu_lepton-MMU),3e-9)
# Finite-T states keep cold composition; they are not finite-T beta solves.
sys.path.insert(0,str(ROOT/'gm1_differential_response'));import adapter
for flavor in ('e','mu'):
 cold=onsets[flavor]['state']
 for T in (2.,8.):
  m=adapter.matter_state(T,cold['nB'],cold['Ye'],cold['Ymu']);delta=m.nuclear.mu_n_mev-m.nuclear.mu_p_mev
  warm.append(dict(cold_onset=flavor,temperature=T,matter=m.asdict(),zeta_e=m.electron.chemical_potential-delta,zeta_mu=m.muon.chemical_potential-delta,scope='Frozen cold composition, no warm beta equilibrium or Urca broadening/rate computed'))
  gate(flavor+' warm preserves Yp'+str(T),abs(m.nuclear.Yp-cold['Yp']),2e-11);gate(flavor+' warm independent charge'+str(T),abs(m.forward_gates['charge_fraction_residual']),2e-10)
 save()
gate('physical sourcehashes unchanged',float(hashes()!=before),.1);save();print(sum(c['passed']for c in checks),'/',len(checks),'runtime',time.monotonic()-start)
for c in checks:
 if not c['passed']:print('FAIL',c)
print('onsets',[(k,v['state']['nB'])for k,v in onsets.items()])
