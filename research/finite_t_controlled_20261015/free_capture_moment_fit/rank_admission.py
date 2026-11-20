from pathlib import Path
import sys,importlib.util,json,hashlib,numpy as np
P=Path(__file__).resolve().parent;sys.dont_write_bytecode=True
s=importlib.util.spec_from_file_location('locked_fit_rank_only',P/'collect.py');m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
lock=json.loads((P/'REVISED_LOCK2.json').read_text());rows=[]
for group in ['training','heldout']:
 for st in lock[group]:
  b,_=m.EC.matched(**st);eta=b.mu_e/b.T;rows.append(dict(group=group,state=st,eta=eta,basis=m.basis(st['T'],st['nB'],eta),admitted=abs(eta)<=3 and max((b.mu_n-m.EC.MN)/b.T,(b.mu_p-m.EC.MP)/b.T)<=-5))
X=np.array([r['basis']for r in rows if r['group']=='training']);sc=np.linalg.norm(X,axis=0);rank=int(np.linalg.matrix_rank(X));out=dict(qualified=rank==7 and all(r['admitted']for r in rows),rank=rank,shape=list(X.shape),raw_condition=float(np.linalg.cond(X)),scaled_condition=float(np.linalg.cond(X/sc)),rows=rows,lock_sha256=hashlib.sha256((P/'REVISED_LOCK2.json').read_bytes()).hexdigest(),rate_data_files_before_rank=list(str(x)for x in P.glob('training_*.json'))+list(str(x)for x in P.glob('heldout_*.json')))
(P/'rank_admission.json').write_text(json.dumps(out,indent=2)+'\n');print(out['qualified'],out['rank'],out['raw_condition'],out['scaled_condition'])
