"""Original total-energy shells and explicit physical/kinetic-current mapping."""
from adapter import *
import json,time
P=Path(__file__).resolve().parent;start=time.monotonic();data=json.loads((P/'results.json').read_text());checks=[];rows=[]
def complex_array(v):return np.array(v['real'])+1j*np.array(v['imag'])if isinstance(v,dict)else np.asarray(v)
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and error<tol)))
for row in data['rows']:
 label=row['label'];s=OrderedState(**row['state']);r=row['runs'][-1];mapping=r['response']['mapping'];omega=mapping['physical_omega'];w=mapping['kinetic_omega'];q=mapping['q'];ff=free.C.FormFactors().values(omega*omega-q*q,s.weak_bare_mass)
 for i,value in enumerate(ff):gate(label+' physicalq2 formfactor '+str(i),abs(value-r['response']['form_factors'][i])/max(abs(value),1.),3e-14)
 entries=[]
 for panel in r['response']['panels'][::max(1,len(r['response']['panels'])//4)]:
  Ea=(panel['lower']+panel['upper'])/2;p=math.sqrt((Ea-s.dirac_mass)*(Ea+s.dirac_mass));x=(w*w-q*q+2*w*Ea)/(2*p*q);Eb=math.sqrt(s.dirac_mass**2+p*p+q*q+2*p*q*x)
  gate(label+' exact total energy delta',abs(omega+(Ea+s.potential_a)-(Eb+s.potential_b)),3e-10);gate(label+' kinetic energy delta',abs(w+Ea-Eb),3e-10);gate(label+' exact supportangle',max(0.,abs(x)-1),3e-12)
  la=float(log_expit((s.total_mu_a-(Ea+s.potential_a))/s.T)+log_expit(((Eb+s.potential_b)-s.total_mu_b)/s.T));lb=float(log_expit((s.total_mu_b-(Eb+s.potential_b))/s.T)+log_expit(((Ea+s.potential_a)-s.total_mu_a)/s.T))
  expected=-(omega+s.total_mu_a-s.total_mu_b)/s.T;gate(label+' original total-energy occupation KMS',abs(lb-la-expected),3e-11)
  entries.append(dict(Ea_star=Ea,Eb_star=Eb,Ea_total=Ea+s.potential_a,Eb_total=Eb+s.potential_b,cosine=x,log_absorption_occupation=la,log_emission_occupation=lb))
 # Physical leptonfourmomenta must not receive nucleon vector shifts.
 k=complex_array(r['lepton_k']).real;l=complex_array(r['lepton_l']).real;ml=r['inputs']['lepton_mass'];gate(label+' physical neutrino massshell',abs(k[0]**2-k[1:]@k[1:])/max(k[0]**2,1.),3e-13);gate(label+' physical chargedlepton massshell',abs(l[0]**2-l[1:]@l[1:]-ml**2)/max(l[0]**2,1.),3e-13)
 rows.append(dict(label=label,shell_samples=entries))
# Reject malformed composition/state before any physical kernel call.
for name,call in [('zero baryon fraction',lambda:matter_state(5.,.2,0.,0.)),('negative density',lambda:matter_state(5.,-.2,.18,.02)),('nan Ye',lambda:matter_state(5.,.2,math.nan,.02)),('zero Dirac mass',lambda:OrderedState(5.,0.,0.,0.,900.,900.,939.)),('nan potential',lambda:OrderedState(5.,600.,math.nan,0.,900.,900.,939.)),('zero weakmass',lambda:OrderedState(5.,600.,0.,0.,900.,900.,0.))]:
 try:call();rejected=False
 except ValueError:rejected=True
 gate('reject '+name,float(not rejected),.1)
(P/'mapping_results.json').write_text(json.dumps(dict(passed=all(c['passed']for c in checks),checks=checks,rows=rows,runtime_seconds=time.monotonic()-start),indent=2)+'\n');print(sum(c['passed']for c in checks),'/',len(checks),'runtime',time.monotonic()-start)
for c in checks:
 if not c['passed']:print('FAIL',c)
