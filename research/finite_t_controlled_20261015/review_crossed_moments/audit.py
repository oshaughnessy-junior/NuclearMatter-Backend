from pathlib import Path
import json,math
from scipy.special import kve
p=Path(__file__).resolve().parent;root=p.parent;checks=[];rows=[]
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
initial=json.loads((root/'crossed_thermal_moments/initial_refinement_failure.json').read_text());gate('original24 failed gates still retained',initial['failed']-24,0.)
for filename in ['results.json','denser_results.json']:
 d=json.loads((p/filename).read_text());inp=d['rows'][0];material=inp['material'];author=next(r['result']for r in d['rows']if r['kind']=='author fresh');T=inp['T'];g=material['neutron'];m=g['mass'];mu=g['chemical_potential'];MB=m*m*T/(math.pi**2*197.3269804**3)*kve(2,m/T)*math.exp((mu-m)/T);difference=MB/g['density']-1
 if filename=='denser_results.json':gate('new denserFD state distinguishes MB approximation',float(difference<1e-4),0.)
 gate(filename+' physical masses',abs(material['neutron']['mass']-939.56542052)+abs(material['proton']['mass']-938.27208816)+abs(material['electron']['mass']-.51099895069),1e-13);gate(filename+' muons explicitly absent',float(material['muons']!='absent'),0.)
 gate(filename+' charge with electron pairs',material['electron']['net_density']/(inp['nB']*inp['Ye'])-1,3e-10)
 for name,s in author['scenarios'].items():
  R=s['net']['number'];sources=s['number_sources'];gate(filename+name+' baryon ledger',sources['neutron']+sources['proton'],0.);gate(filename+name+' charge ledger',sources['proton']-sources['electron_net'],0.);gate(filename+name+' electron lepton ledger',sources['electron_net']-sources['antineutrino_e'],0.)
 rows.append(dict(filename=filename,FD_neutron_density=g['density'],MB_same_mu_density=MB,MB_relative_density_excess=difference,neutron_pair_logratio=g['log_antiparticle_to_particle_ratio'],electron_particle_density=material['electron']['particle_density'],electron_antiparticle_density=material['electron']['antiparticle_density']))
(p/'audit_results.json').write_text(json.dumps(dict(passed=all(c['passed']for c in checks),checks=checks,rows=rows),indent=2)+'\n');print(len(checks),all(c['passed']for c in checks))
