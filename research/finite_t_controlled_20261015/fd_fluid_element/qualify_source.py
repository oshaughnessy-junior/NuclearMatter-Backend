"""FD derivative/energy inversion checks and bounded initial physical-source gate."""
from element import *
import qualification_contract as contract
expected=contract.fingerprint(2.,1e-4,.3)
import json
P=Path(__file__).resolve().parent;checks=[]
def check(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(error<tol)))
for T,Y,photons in [(2.,.3,False),(2.2,.28,False),(2.,.3,True)]:
 s=eos(T,1e-4,Y,photons=photons)
 for h in (1e-4,5e-5):
  dT=(eos(T+h,1e-4,Y,photons=photons)['u_MeV_per_baryon']-eos(T-h,1e-4,Y,photons=photons)['u_MeV_per_baryon'])/(2*h)
  dY=(eos(T,1e-4,Y+h,photons=photons)['u_MeV_per_baryon']-eos(T,1e-4,Y-h,photons=photons)['u_MeV_per_baryon'])/(2*h)
  check(f'uT FD T{T}Y{Y}photons{photons}h{h}',abs(dT/s['u_T']-1),3e-8)
  check(f'uY FD T{T}Y{Y}photons{photons}h{h}',abs(dY/s['u_Y']-1),5e-8)
 check(f'energy inverse photons{photons}T{T}Y{Y}',abs(recover_temperature(s['u_MeV_per_baryon'],1e-4,Y,1.1*T,photons=photons)/T-1),2e-11)
start=time.monotonic();coarse=source(2.,1e-4,.3,outer_order=6);fine=source(2.,1e-4,.3,outer_order=10)
accepted=json.loads((ROOT/'quantum_energy_moments'/'summary.json').read_text())
for flavor in ('nue','antinue'):
 ref=next(x['values'] for x in accepted['species'] if x['state']=='dilute' and x['flavor']==flavor)
 for field,key in [('number_fm3_s','emission_number_cm3_s'),('energy_MeV_fm3_s','emission_energy_MeV_cm3_s')]:
  check(f'{flavor} outer6->10 {field}',abs(coarse['moments'][flavor][field]/fine['moments'][flavor][field]-1),3e-6)
  check(f'{flavor} new100MeV panels versus accepted320MeV {field}',abs(fine['moments'][flavor][field]/(ref[key]*1e-39)-1),3e-6)
check('initial heating sign',float(fine['dT_dt_no_photons']<=0),.1)
report=dict(passed=all(x['passed'] for x in checks),checks=checks,coarse_source=coarse,fine_source=fine,runtime_seconds=time.monotonic()-start,scope='Initial physical source plus independent FD derivatives; no trajectory yet')
if expected!=contract.fingerprint(2.,1e-4,.3):raise RuntimeError('Qualification source changed during evaluation')
report=contract.stamp(report,expected)
(P/'source_qualification.json').write_text(json.dumps(report,indent=2)+'\n');print(sum(c['passed'] for c in checks),'/',len(checks),report['runtime_seconds']);print('dT',fine['dT_dt_no_photons'],'dY',fine['dYe_dt'])
for c in checks:
 if not c['passed']:print('FAIL',c)
