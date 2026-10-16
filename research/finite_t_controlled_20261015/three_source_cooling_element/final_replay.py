"""One final guarded fullHeun4 replay, preserving original convergence evidence."""
from pathlib import Path
import importlib.util,sys,json
P=Path(__file__).parent;sp=importlib.util.spec_from_file_location('three_source_final_element_functions',P/'element.py');E=importlib.util.module_from_spec(sp);sys.modules[sp.name]=E;sp.loader.exec_module(E);old=json.loads((P/'results.json').read_text())['rows'][-1];new=E.trajectory('Heun',4,.002)
def strip(x):
 if isinstance(x,dict):return{k:strip(v)for k,v in x.items()if k not in['seconds','elapsed']}
 if isinstance(x,list):return list(map(strip,x))
 return x
same=strip(old)==strip(new);out=dict(final_source_identity=E.M.Client().identity,whole_physical_trajectory_bitwise_same=same,trajectory=new,passed=int(same),failed=[]if same else['finalguard replay differs']);(P/'final_replay.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(passed=out['passed'],failed=out['failed'],seconds=new['seconds']),indent=2))
