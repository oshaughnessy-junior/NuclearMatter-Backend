from pathlib import Path
import json,math,hashlib
P=Path(__file__).resolve().parent
states=[]
for i in range(3):
 p=P/f'state_{i}.json'
 if not p.exists():continue
 d=json.loads(p.read_text());s={k:d.get(k)for k in['T','nB','status','passed','failed','seconds','kinetic_Ye','chemical_Ye','exception']}
 if 'balanced'in d:
  r=d['balanced'];s.update(affinity_MeV=r['chemical_affinity'],rates=r['rates'],energy=r['energy'],escaping_energy=r['escaping_energy'],normalized_drift=r['net_material_number_source']/sum(r['rates'].values()),chemical_dYe_dt=d['chemical_source']['dYe_dt']if 'chemical_source'in d else None,max_source_refinement=max([abs(c['error'])for c in d['checks']if'quadrature'in c['name']]+[0]),max_isolated_tail=max([abs(c['error'])for c in d['checks']if'isolated tail'in c['name']]+[0]),actual_source_trial_pairs=len(d['rows']))
 states.append(s)
pins=json.loads((P/'SOURCE_PIN.json').read_text());unchanged=all(hashlib.sha256((P.parent/n).read_bytes()).hexdigest()==h for n,h in pins.items())
out=dict(states=states,source_original_anchor_hashes_unchanged=unchanged,independent_review='PENDING',scope='Onlythree selected new prescribed baths; actual freebeta+PC+EC with empty incoming neutrinos, fixedT thermostat. No fit/interactingEOS/fullweak/hydro/uncertainty/AD claim.')
(P/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
