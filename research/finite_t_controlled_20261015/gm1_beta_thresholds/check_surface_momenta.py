"""Reconstruct baryon momentum densities from chemical potentials, not input n."""
from cold_beta import *
import json
P=Path(__file__).resolve().parent;a=json.loads((P/'results.json').read_text());b=json.loads((P/'pinned_results.json').read_text());checks=[];rows=[]
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and error<tol)))
for flavor in ('e','mu'):
 s=a['onsets'][flavor]['state'];n=s['nuclear'];M=n['dirac_mass_mev'];Un=n['omega_mev']-.5*n['rho_mev'];Up=n['omega_mev']+.5*n['rho_mev'];record=dict(flavor=flavor,reconstructed=[]);rows.append(record)
 for name,mu,U,target in [('neutron',n['mu_n_mev'],Un,s['nB']*(1-s['Yp'])),('proton',n['mu_p_mev'],Up,s['nB']*s['Yp'])]:
  nu=mu-U;p=math.sqrt((nu-M)*(nu+M));measured=quad(lambda k:k*k,0,p,epsabs=1e-20,epsrel=2e-12)[0]/(math.pi**2*H**3);gate(flavor+' Fermi surface density '+name,abs(measured/target-1),3e-11);record['reconstructed'].append(dict(species=name,kinetic_mu=nu,pF_MeV=p,momentum_density=measured,target=target))
 # Actual pinned composition is available at each numerical onset's earlier bracketing estimate.
 old=min(b['rows'],key=lambda r:abs(r['nB']-s['nB']));v=old['pinned_source'];nb=old['nB'];yl=v['Ye']if flavor=='e'else v['Ymu'];delta=H*((3*math.pi**2*nb*(1-v['Yp']))**(1/3)-(3*math.pi**2*nb*v['Yp'])**(1/3)-(3*math.pi**2*nb*yl)**(1/3));gate(flavor+' actualpinnedsource atonset triangle',abs(delta),3e-8);record['pinned_triangle_delta_MeV']=delta
(P/'surface_results.json').write_text(json.dumps(dict(passed=all(c['passed']for c in checks),checks=checks,rows=rows),indent=2)+'\n');print(sum(c['passed']for c in checks),'/',len(checks))
