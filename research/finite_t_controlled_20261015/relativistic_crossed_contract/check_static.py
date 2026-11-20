from pathlib import Path
import sys,json,math
sys.dont_write_bytecode=True
from beta import *
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'weak_rates'));import static_capture as W
base=next(r for r in json.loads(Path(__file__).with_name('results.json').read_text())['rows']if 'static_rate'in r)
old=W.target_moments(.001,0,'neutron_decay',order=96,mass=ME,q=MN-MP);error=old[0]/base['static_rate']-1
out=dict(passed=bool(abs(error)<2e-12),checks=[dict(name='older static reference exact matched vacuum limit',error=float(error),tolerance=2e-12,passed=bool(abs(error)<2e-12))],older_static_rate=float(old[0]),older_neutrino_energy_moment=float(old[1]),new_static_rate=base['static_rate'],T=.001,mu_e=0,blocking_bound=math.exp(-ME/.001))
Path(__file__).with_name('static_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out['passed'],error)
