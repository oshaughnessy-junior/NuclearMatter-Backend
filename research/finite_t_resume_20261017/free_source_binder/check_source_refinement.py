"""Selected changed-state base/fine channel profile gates; actual quadrature."""
from pathlib import Path
from dataclasses import asdict
import json,signal,time,math
from nmbackend.reference import free_matter as F
from nmbackend.reference import weak_reference as W
P=Path(__file__).resolve().parent
signal.alarm(88);start=time.monotonic();fine=json.loads((P/'source_1.json').read_text());checks=[];rows=[]
s=F.state_density(2.03,1.05e-7,.34,photons='trapped_LTE')
bath=W.FreeElectronBath(s.temperature_mev,s.neutron.chemical_potential,s.proton.chemical_potential,s.electron.chemical_potential)
for channel,ref in zip(W.CHANNELS,fine['channels']):
 r=W.channel_moments(channel,bath,W.Radiation.empty(),W.Controls.base(channel));rows.append(asdict(r))
 for i,(q,f) in enumerate(zip(r.forward,ref['forward'])):
  error=math.expm1(q.log-f['log']);checks.append(dict(name=channel+' forward '+r.names[i],error=error,tolerance=3e-6,passed=abs(error)<=3e-6))
checks.append(dict(name='fresh changed-state bath matches fine',error=0.,tolerance=0.,passed=asdict(bath)==fine['channels'][0]['bath']))
out=dict(checks=checks,passed=sum(x['passed'] for x in checks),failed=[x for x in checks if not x['passed']],base_channel_records=rows,fine_record='source_1.json',seconds=time.monotonic()-start,qualification='selected base/fine comparisons only; not an independent absolute physical-model oracle or uniform convergence certificate')
(P/'source_refinement_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ['passed','failed','seconds']}));signal.alarm(0)
