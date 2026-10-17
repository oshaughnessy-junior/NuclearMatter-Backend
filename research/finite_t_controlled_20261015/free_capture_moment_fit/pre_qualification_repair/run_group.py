from pathlib import Path
import importlib.util,sys,json,time
P=Path(__file__).resolve().parent;sys.dont_write_bytecode=True
s=importlib.util.spec_from_file_location('locked_collection_group',P/'collect.py');m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
group=sys.argv[1]
if group not in['training','heldout']:raise ValueError('Declared group')
if group=='heldout'and not(P/'COEFFICIENT_FREEZE.json').exists():raise RuntimeError('Fit must be frozen before heldout collection')
start=time.monotonic();rows=[]
for i in range(len(json.loads((P/'REVISED_LOCK2.json').read_text())[group])):
 if time.monotonic()-start>110:raise TimeoutError('Group bound; per-state checkpoints retained')
 r=m.collect(group,i);rows.append(dict(index=i,qualified=r['qualified'],seconds=r['seconds']))
 print(rows[-1],flush=True)
 if not r['qualified']:raise ArithmeticError('Actual source refinement failed; state kept, no fit admission')
(P/(group+'_collection_complete.json')).write_text(json.dumps(dict(rows=rows,seconds=time.monotonic()-start),indent=2)+'\n')
