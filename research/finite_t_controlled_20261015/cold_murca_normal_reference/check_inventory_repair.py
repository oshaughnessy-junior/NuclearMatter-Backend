"""Independent count/unit regression; no physical rate/geometry evaluation."""
from pathlib import Path
import json,hashlib,math
P=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
old=json.loads((P/'pre_inventory_repair/reaction_inventory.json').read_text());new=json.loads((P/'reaction_inventory.json').read_text());checks=[]
def check(name,ok):checks.append(dict(name=name,passed=bool(ok)))
check('all eight directed events retained',len(old)==len(new['events'])==8)
for a,b in zip(old,new['events']):
    name=b['id']
    check(name+' original integer values unchanged',a['delta_material_Ye_per_event']==b['delta_Ne_per_event'] and a['delta_material_Ymu_per_event']==b['delta_Nmu_per_event'] and a['delta_proton']==b['delta_Np_per_event'] and a['delta_neutron']==b['delta_Nn_per_event'])
    check(name+' electric charge',b['delta_Np_per_event']-b['delta_Ne_per_event']-b['delta_Nmu_per_event']==0)
    check(name+' baryon number',b['delta_Np_per_event']+b['delta_Nn_per_event']==0)
    check(name+' flavor lepton numbers',b['delta_Ne_per_event']+b['delta_radiation_Le_per_event']==0 and b['delta_Nmu_per_event']+b['delta_radiation_Lmu_per_event']==0)
# Prescribed synthetic NUMBER rates, expressly not computed MU emissivities.
R={'neutron_electron_beta_decay':3e-10,'proton_electron_capture':1e-10,'proton_muon_beta_decay':2e-10,'neutron_muon_capture':.5e-10};nB=1e-4
def dot(k):return sum(e[k]*R.get(e['id'],0.) for e in new['events'])
fraction={i:dot('delta_N'+i+'_per_event')/nB for i in ['e','mu','p','n']}
want=dict(e=2e-6,mu=1.5e-6,p=3.5e-6,n=-3.5e-6)
for i,v in want.items():check('count-rate to fraction-rate '+i,math.isclose(fraction[i],v,rel_tol=2e-15))
V=1e8;NB=nB*V
check('single beta event fraction change requires NB',new['events'][0]['delta_Ne_per_event']/NB==1e-4)
check('density doubling halves fraction source',math.isclose(dot('delta_Ne_per_event')/(2*nB),fraction['e']/2,rel_tol=1e-15))
check('independent finite-volume count ledger',math.isclose(dot('delta_Np_per_event')*V/NB,fraction['p'],rel_tol=1e-15))
check('no ambiguous fraction-named integers remain',all(not any('delta_Y' in k or 'delta_material_Y' in k for k in e) for e in new['events']))
pins=json.loads((P/'PRE_INVENTORY_REPAIR_PIN.json').read_text())
for f in ['normal_reference.py','angular_oracle.py','validate_reference.py','results.json']:
    check('scientific source/raw unchanged '+f,sha(P/f)==pins[f])
out=dict(checks=checks,passed=sum(x['passed'] for x in checks),failed=[x for x in checks if not x['passed']],synthetic_number_rates_fm3_s=R,nB_fm3=nB,fraction_rates_s_inverse=fraction,original_snapshot=pins,numerical_rate_rerun=False)
(P/'inventory_repair_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ['passed','failed']}))
