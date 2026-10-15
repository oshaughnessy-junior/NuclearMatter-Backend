"""Independent Fermi-momentum geometry for cold finite-E vs zero-E onsets."""
from pathlib import Path
import sys,json,math,time
from scipy.optimize import brentq
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'gm1_beta_thresholds'));import cold_beta as B
start=time.monotonic();d=json.loads((ROOT/'gm1_threshold_thermal_probes/results.json').read_text());checks=[]
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
def geometry(n,E):
 s=B.state(n);M=s.nuclear.dirac_mass_mev;pn=B.H*(3*math.pi**2*n*(1-s.Yp))**(1/3);pp=B.H*(3*math.pi**2*n*s.Yp)**(1/3);pe=s.electron['kF_MeV'];nu_p=math.hypot(M,pp);excited=math.sqrt((nu_p+E)**2-M*M)
 # At maximal incoming cold neutron energy, outgoing proton has nu_p+Enu.
 # Backscattered physical lepton pair has q=pe+Enu, including Enu momentum.
 return pn-excited-pe-E
zero=brentq(lambda n:geometry(n,0),.26,.30,xtol=4e-13,rtol=3e-14);finite=brentq(lambda n:geometry(n,.3),.26,.30,xtol=4e-13,rtol=3e-14)
gate('zero-neutrino triangle independent geometry',zero-d['cold_onset']['nB'],3e-11);gate('finite-Enu selected probe independent geometry',finite-d['finite_incident_selected_probe_cold_onset'],3e-11);gate('finite onset genuinely distinct from zero limit',float(finite>=zero),0.)
for i,r in enumerate(d['rows']):
 expected=r['density_factor']==.97
 gate(f'row{i} cold reference opening label',float(r['thermal_opening_of_cold_reference']!=expected),0.)
 gate(f'row{i} instantaneous Hamiltonian opening label',float(r['thermal_opening_at_same_instantaneous_Hamiltonian']!=expected),0.)
 s=r['state'];aff=r['electron']['chemical_potential']+s['total_mu_b']-s['total_mu_a'];gate(f'row{i} actual affinity retained',aff-r['affinities']['e'],3e-12)
 gate(f'row{i} bare939',s['weak_bare_mass']-939.,0.)
out=dict(passed=all(c['passed']for c in checks),checks=checks,zero_neutrino_onset=zero,finite_Enu03_selected_onset=finite,shift=finite-zero,runtime_seconds=time.monotonic()-start)
(HERE/'onset_results.json').write_text(json.dumps(out,indent=2)+'\n');print(len(checks),out['passed'],out['runtime_seconds'])
