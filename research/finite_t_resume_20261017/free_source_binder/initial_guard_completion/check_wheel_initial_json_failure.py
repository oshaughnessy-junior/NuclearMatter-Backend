"""Fresh unpacked-wheel import and actual source; no installation or mock kernels."""
from pathlib import Path
from dataclasses import asdict
import json,sys,time,signal,hashlib
import numpy,scipy,jax,nmbackend
from nmbackend.reference import free_matter as F
from nmbackend.reference import weak_reference as W
P=Path(__file__).resolve().parent
signal.alarm(88);start=time.monotonic()
assert 'wheel_unpacked' in F.__file__
s=F.state_density(2.,1e-7,.3,photons='absent')
source=F.source_moments(s,radiation=W.Radiation.empty(),controls=F.SourceControls.fine())
rhs=F.instantaneous_rhs(s,source,baryon_density_dot_fm3_s=2.5e-8)
reference=json.loads((P/'source_0.json').read_text())
checks=[]
for label,value,expected in [('R',source.proton_number_source.value,reference['R']['value']),('Q',source.neutrino_energy_escape.value,reference['Q']['value']),('Tdot',rhs.temperature_dot_mev_s,reference['rhs']['temperature_dot_mev_s'])]:
 error=abs(value/expected-1);checks.append(dict(name=label,error=error,tolerance=2e-12,passed=error<=2e-12))
origins={m.__name__:m.__file__ for m in [nmbackend,F,W]}
report=dict(checks=checks,passed=sum(c['passed'] for c in checks),failed=[c for c in checks if not c['passed']],origins=origins,python=sys.version,nmbackend_version=nmbackend.__version__,numpy=numpy.__version__,scipy=scipy.__version__,jax=jax.__version__,source_state=asdict(s),source_identity=source.identity,R=asdict(source.proton_number_source),Q=asdict(source.neutrino_energy_escape),rhs=asdict(rhs),seconds=time.monotonic()-start,qualification='Offline wheel built with existing backend, unpacked/imported in fresh process from /private/tmp; not a clean dependency installation/minimum-version test')
(P/'wheel_result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:report[k] for k in ['passed','failed','origins','seconds']}));signal.alarm(0)
