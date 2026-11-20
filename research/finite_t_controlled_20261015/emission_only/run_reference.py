"""Bounded heavy-M complete named antineutrino inventory qualification."""
from pathlib import Path
import sys,json,math,time,hashlib
import numpy as np
from scipy.special import logsumexp
import log_emission as r

HERE=Path(__file__).resolve().parent;start=time.monotonic();checks=[];runs=[];failures=[]
def gate(name,error,tol):
    checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and error<tol)))

bath=r.make_bath(.3,7e-5,3e-5,1e7,0.)
state=dict(T_MeV=bath.T,mass_MeV=bath.neutron.mass,mu_e_prescribed_MeV=bath.mu_e,neutron=bath.neutron.asdict(),proton=bath.proton.asdict(),charge_neutrality_imposed=False)
for name,sp in [('neutron',bath.neutron),('proton',bath.proton)]:
    thermal=2*(sp.mass*sp.T/(2*math.pi))**1.5/r.static.HBARC**3
    z=math.exp(sp.kinetic_mu/sp.T)
    series=thermal*sum((-1)**(k+1)*z**k/k**1.5 for k in range(1,9))
    gate(name+' independent FD fugacity-series density',abs(series/sp.density-1),1e-9)
    state[name]['classical_density_over_actual_minus1']=thermal*z/sp.density-1
    state[name]['fugacity']=z

expected={c:bath.neutron.density*r.static.target_moments(bath.T,bath.mu_e,c,192)for c in r.CHANNELS}
expected['combined']=expected['positron_capture']+expected['neutron_decay']
paths=[Path(r.__file__),Path(r.eos.__file__),Path(r.guarded_capture.__file__),Path(r.crossed_decay.__file__),r.ROOT/'capture_endpoint/mapped_capture.py',r.ROOT/'response_reference/free_density_response.py',r.ROOT/'weak_rates/static_capture.py']
hashes={str(p.relative_to(r.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths}
original=r.ROOT/'quantum_energy_moments/static_inventory_failure.json'
original_hash=hashlib.sha256(original.read_bytes()).hexdigest()
try:
    for order in [8,12,20]:
        remaining=110-(time.monotonic()-start)
        if remaining<=0:raise TimeoutError('110second total experiment bound reached')
        result=r.log_moments(bath,outer_order=order,internal_order=24,angle_order=24,max_seconds=remaining)
        runs.append(result)
        (HERE/'partial.json').write_text(json.dumps(dict(runs=runs,elapsed_seconds=time.monotonic()-start),indent=2)+'\n')
    fine=runs[-1]
    for channel in [*r.CHANNELS,'combined']:
        logs=fine['log_moments'][channel]
        for k,name in enumerate(['number','energy']):
            gate(f'absolute heavyM {channel} {name}',abs(math.expm1(logs[k]-math.log(expected[channel][k]))),3e-5)
            for coarse in runs[:-1]:
                gate(f'outer{coarse["outer_order"]}->20 {channel} {name}',abs(math.expm1(coarse['log_moments'][channel][k]-logs[k])),3e-6)
            for cutoff,tol in [(12.,3e-6),(18.,3e-10)]:
                if channel=='combined':
                    panel_logs=[float(logsumexp([p['log_number_energy'][c][k]for c in r.CHANNELS]))for p in fine['panels']if p['hi']<=cutoff]
                else:panel_logs=[p['log_number_energy'][channel][k]for p in fine['panels']if p['hi']<=cutoff]
                gate(f'outer tail cutoff{cutoff}->30 {channel} {name}',abs(math.expm1(float(logsumexp(panel_logs))-logs[k])),tol)
    # Independently vary internal quadrature/tail at selected allowed peaks.
    for E,channel in [(.1,'neutron_decay'),(.4,'neutron_decay'),(.7,'neutron_decay'),(2.1,'positron_capture'),(4.,'positron_capture')]:
        coarse=r.log_spectra(bath,E,energy_order=24,angle_order=24)
        refined=r.log_spectra(bath,E,energy_order=36,angle_order=36,tail=60.)
        gate(f'internal24->36tail60 {channel} E{E}',abs(math.expm1(refined[channel]-coarse[channel])),3e-6)
    # The gap witness is retained logarithmically; ordinary values are unavailable.
    witness_logs=r.log_spectra(bath,.8313293913178066)
    witness={c:r.represented(v)for c,v in witness_logs.items()}
    gate('gap witness log coefficients remain finite',0. if all(math.isfinite(v)for v in witness_logs.values())else 1.,.5)
    gate('gap witness ordinary combined flagged unavailable',0. if not witness['combined']['representable']and witness['combined']['value']is None else 1.,.5)
except Exception as exc:
    failures.append(dict(type=type(exc).__name__,message=str(exc)))
    witness=None

final={str(p.relative_to(r.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths}
report=dict(status='PASS'if not failures and all(c['passed']for c in checks)and hashes==final else'FAIL',checks=checks,failures=failures,state=state,runs=runs,static_expected={c:v.tolist()for c,v in expected.items()},gap_witness=witness,source_sha256=hashes,end_source_sha256=final,original_failure_file=str(original.relative_to(r.ROOT)),original_failure_sha256=original_hash,original_failure_unchanged=original_hash==hashlib.sha256(original.read_bytes()).hexdigest(),runtime_seconds=time.monotonic()-start,
            scope='One prescribed-electron heavy-commonM FD-density-matched antineutrino complete NAMED inventory, mathematical static normalization limit; no opacity or general domain certificate')
(HERE/'results.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(status=report['status'],checks=len(checks),failures=failures,failed=[c for c in checks if not c['passed']],runtime=report['runtime_seconds']),indent=2))
raise SystemExit(0 if report['status']=='PASS'else 1)
