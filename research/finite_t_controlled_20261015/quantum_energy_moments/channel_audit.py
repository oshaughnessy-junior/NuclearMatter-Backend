"""Resolve the actual antineutrino inventory on completed outer8 nodes."""
from integrated import *
import json,time,sys
P=Path(__file__).resolve().parent;tag=sys.argv[1];d=json.loads((P/(tag+'_results.json')).read_text());state=d['state']
m=slab.matter_state(state['T'],state['nn'],state['np']);r=next(r for r in d['runs'] if r['flavor']=='antinue' and r['outer_order']==8)
sums={k:np.zeros(2) for k in ('capture','crossed_neutron_decay')};start=time.monotonic();guards=0
for n in r['nodes']:
 E=n['E'];c=slab.coefficients(m,'antinue',E,order=48)
 guards+=int(c['capture_panel_guard']['used'])
 for k,channel in c['channels'].items():
  number=n['weight']*slab.DOS*E*E*1e39*math.exp(channel['log_j'])
  sums[k]+=np.array([number,E*number])
total=sum(sums.values());expected=np.array([r['values']['emission_number_cm3_s'],r['values']['emission_energy_MeV_cm3_s']]);errors=abs(total/expected-1)
report=dict(tag=tag,channel_moments={k:dict(number_cm3_s=float(v[0]),energy_MeV_cm3_s=float(v[1])) for k,v in sums.items()},sum_relative_errors=errors.tolist(),passed=bool(np.all(errors<3e-13)),guard_count=guards,runtime_seconds=time.monotonic()-start,qualification='Inventory sum audit at already refined outer8 nodes; not separate per-channel continuum convergence')
(P/(tag+'_channel_audit.json')).write_text(json.dumps(report,indent=2)+'\n');print(tag,'channel audit',report['passed'],report['runtime_seconds'])
