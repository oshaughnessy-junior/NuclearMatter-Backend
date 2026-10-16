"""Synthetic node-contract attacks; never change actual physical raw data."""
from pathlib import Path
import json,importlib.util,sys,copy,math
P=Path(__file__).parent;sp=importlib.util.spec_from_file_location('hot_EC_assembly_guard_functions',P/'total.py');T=importlib.util.module_from_spec(sp);sys.modules[sp.name]=T;sp.loader.exec_module(T);x=json.loads((P/'total_fine_part0.json').read_text());c=x['contract'];r=x['rows'][0];checks=[]
def ck(n,v):checks.append(dict(name=n,passed=bool(v)))
T.validate_node(c,r);ck('actual exact complete node accepted',True)
attacks=[('controls',lambda a:a['spectrum']['controls'].update(angle_order=4)),('bath',lambda a:a['spectrum']['bath'].update(mu_e=.175)),('state',lambda a:a['spectrum']['matched_state'].update(Ye=.31)),('physicalsource',lambda a:a['spectrum']['source_pins'].update({'relativistic_tensor/tensor.py':'0'*64})),('implementation',lambda a:a['spectrum'].update(implementation_sha256='0'*64)),('units',lambda a:a['spectrum'].update(units='cm^-1')),('channel',lambda a:a['spectrum'].update(channel='all')),('boolindex',lambda a:a.update(index=False)),('node',lambda a:a.update(E=a['E']+.01)),('weight',lambda a:a.update(weight=a['weight']*1.01)),('falsezero',lambda a:a['spectrum']['j'].update(value=0.,log=None,representable=True,reason='empty_support')),('underflow',lambda a:a['spectrum']['j'].update(value=None,log=-800.,representable=False,reason='underflow')),('floor',lambda a:a['spectrum']['j'].update(value=1e-200)),('nonfinite',lambda a:a['spectrum']['chi'].update(value=float('nan'))),('boolvalue',lambda a:a['spectrum']['chi'].update(value=True))]
for name,f in attacks:
 a=copy.deepcopy(r);f(a)
 try:T.validate_node(c,a);ok=False
 except(RuntimeError,ValueError,TypeError):ok=True
 ck('reject synthetic '+name,ok)
old=json.loads((P/'pre_assembly_guard_total_results.json').read_text());new=json.loads((P/'total_results.json').read_text());ck('assembly guard no numerical result change',new==old)
out=dict(checks=checks,passed=sum(v['passed']for v in checks),failed=[v for v in checks if not v['passed']],scope='synthetic metadata/representation only; no real raw mutation');(P/'assembly_guard_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(passed=out['passed'],failed=out['failed']),indent=2))
