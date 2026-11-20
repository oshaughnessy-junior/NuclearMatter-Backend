"""Check serialized matched-time energy, heat and composition refinements."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

here = Path(__file__).resolve().parent
path = here/'results.json'
r = json.loads(path.read_text())
errors = []
for case in r['cases']:
    for a, b in ((case['runs'][0],case['runs'][1]),(case['runs'][2],case['runs'][3])):
        differences = {}
        for j, label in ((1,'Ye'),(2,'escaped_energy'),(3,'thermostat_heat')):
            scale = max(1.,max(abs(row[j]) for row in b['samples']))
            differences[label] = max(abs(x[j]-y[j]) for x,y in zip(a['samples'],b['samples']))/scale
        assert max(differences.values()) < 1e-6
        errors.append({'T_MeV':case['T_MeV'],'nB_fm-3':case['nB_fm-3'],
                       'initial_Ye':a['initial_Ye'],'scaled_coarse_vs_fine_history_errors':differences})
out = {'status':'PASS','command':f'{sys.executable} {Path(__file__).resolve()}',
       'results_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
       'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       'comparison':'Matched serialized sample times; scales1Ye or1MeV/baryon and gross-integral magnitude',
       'history_refinement':errors}
(here/'refinement_summary.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'status':'PASS','histories_checked':len(errors),
                  'max_scaled_difference':max(v for e in errors for v in e['scaled_coarse_vs_fine_history_errors'].values())},indent=2))
