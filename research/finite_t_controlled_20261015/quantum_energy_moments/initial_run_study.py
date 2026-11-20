"""Independent outer resolution/tail ladders and sampled internal refinement."""
from pathlib import Path
import json,sys,time,math,hashlib
import numpy as np
from integrated import *
P=Path(__file__).resolve().parent
tag=sys.argv[1] if len(sys.argv)>1 else 'dilute'
nn,np_={'dilute':(7e-5,3e-5),'moderate':(.002,.001)}[tag]
m=slab.matter_state(2.,nn,np_)
checks=[];runs=[];refinements=[];start=time.monotonic()
cachefile=P/(tag+'_partial.json')
# Reproduction is fresh by default. Explicit resume only for an interrupted
# run with the same physics and controls; never silently reuse old outputs.
cached=json.loads(cachefile.read_text())['runs'] if '--resume' in sys.argv[2:] and cachefile.exists() else []
def check(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(error<tol)))
# Independent numerical dimensions: fixed internal48/tail40 outer8,12,20;
# then weighted representative nodes internal72/tail50, not all nodes.
for flavor in ('nue','antinue'):
    triplet=[]
    for order in (8,12,20):
        matches=[r for r in cached if r['flavor']==flavor and r['outer_order']==order]
        r=matches[0] if matches else integrate(m,flavor,outer_order=order)
        triplet.append(r);runs.append(r)
        (P/(tag+'_partial.json')).write_text(json.dumps(dict(runs=runs,elapsed_seconds=time.monotonic()-start),indent=2)+'\n')
    fine=triplet[-1]
    for coarse in triplet[:-1]:
        for key in NAMES:
            check(f'{flavor} outer{coarse["outer_order"]}->20 {key}',abs(coarse['values'][key]/fine['values'][key]-1),2e-5)
    for cutoff in (120.,180.,240.):
        truncated=np.sum([p['values'] for p in fine['panels'] if p['hi']<=cutoff],axis=0)
        full=np.array([fine['values'][k] for k in NAMES])
        for k,a,b in zip(NAMES,truncated,full):check(f'{flavor} cutoff{cutoff}->320 {k}',abs(a/b-1),3e-4 if cutoff==120 else (2e-7 if cutoff==180 else 2e-10))
    v=fine['values'];gross=v['slab_emission_number_cm2_s']+v['slab_absorbed_boundary_number_cm2_s']
    check(flavor+' independent integrated number flux balance',abs(v['outgoing_left_number_cm2_s']+v['outgoing_right_number_cm2_s']-v['incoming_left_number_cm2_s']-v['incoming_right_number_cm2_s']-v['radiation_number_gain_cm2_s'])/gross,1e-12)
    grosse=v['slab_emission_energy_MeV_cm2_s']+v['slab_absorbed_boundary_energy_MeV_cm2_s']
    check(flavor+' independent integrated energy flux balance',abs(v['outgoing_left_energy_MeV_cm2_s']+v['outgoing_right_energy_MeV_cm2_s']-v['incoming_left_energy_MeV_cm2_s']-v['incoming_right_energy_MeV_cm2_s']-v['radiation_energy_gain_MeV_cm2_s'])/grosse,1e-12)
    # Max-weight node per energy panel, including very-low E, both sides of
    # physical thresholds, bulk emissivity and high-energy tail.
    for panel in fine['panels']:
        candidates=[n for n in fine['nodes'] if panel['lo']<n['E']<panel['hi']]
        n=max(candidates,key=lambda n:n['weight']*n['values'][1])
        E=n['E'];base=np.array(n['values']);better,c=spectral(m,flavor,E,1e5,72,50.)
        errs=np.abs(better-base)/np.maximum(np.abs(better),1e-300)
        for k,error in zip(NAMES,errs):check(f'{flavor} internal48->72 E{E:.8g} {k}',error,3e-5)
        check(f'{flavor} sampled KMS E{E:.8g}',abs(c['log_j']-c['log_chi']+(E-c['mu_neutrino_MeV'])/m.neutron.T),2e-10)
        refinements.append(dict(flavor=flavor,E=E,base=base.tolist(),refined=better.tolist(),relative_errors=errs.tolist(),mu_neutrino=c['mu_neutrino_MeV']))
    # Independent FD incoming quadrature includes prescribed +/- mu signs.
    from scipy.integrate import quad
    for side,(T,mu,amplitude) in enumerate([(4.,3.,1.),(6.,-2.,.35)]):
        for power,kind in [(2,'number'),(3,'energy_MeV')]:
            expected=slab.LIGHT*slab.DOS*1e39/4*amplitude*quad(lambda E:E**power*float(slab.expit((mu-E)/T)),0.,500.,epsabs=1e-8,epsrel=1e-12)[0]
            key=f'incoming_{["left","right"][side]}_{kind}_cm2_s'
            check(f'{flavor} independent signedmu FD {key}',abs(v[key]/expected-1),3e-10)
report=dict(tag=tag,state=dict(T=m.neutron.T,nn=nn,np=np_,mu_e=m.electron.chemical_potential,mu_nue=m.optical('nue').mu_neutrino),passed=all(c['passed'] for c in checks),checks=checks,runs=runs,internal_refinements=refinements,runtime_seconds=time.monotonic()-start,cached_integration_seconds=sum(r['runtime_seconds'] for r in cached),scope='Full energy quadrature over free NR FD charged-current ansatz. Sampled, not global, internal-grid qualification. No fluid evolution.')
(P/(tag+'_results.json')).write_text(json.dumps(report,indent=2)+'\n')
print(tag,sum(c['passed'] for c in checks),'/',len(checks),'runtime',report['runtime_seconds'])
for c in checks:
    if not c['passed']:print('FAIL',c)
