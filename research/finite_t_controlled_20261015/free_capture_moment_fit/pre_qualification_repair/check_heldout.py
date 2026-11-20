from pathlib import Path
import json,hashlib,numpy as np
P=Path(__file__).resolve().parent;fit=json.loads((P/'COEFFICIENT_FREEZE.json').read_text());lock=json.loads((P/'REVISED_LOCK2.json').read_text());checks=[];rows=[]
for i in range(8):
 raw=json.loads((P/f'heldout_{i}.json').read_text())
 if raw['qualified']is not True:raise ArithmeticError('Unqualified heldout physical reference')
 pred=np.asarray(raw['allowed'])*np.exp(np.asarray(raw['basis'])@np.asarray(fit['coefficients']));ref=np.asarray(raw['refined']['values']);err=pred/ref-1
 for k,name in enumerate(fit['outputs']):checks.append(dict(name=f'heldout{i} {name}',error=float(err[k]),gate=.01,passed=bool(abs(err[k])<=.01)))
 rows.append(dict(index=i,state=raw['state'],predicted=pred.tolist(),actual=ref.tolist(),relative_errors=err.tolist(),raw_sha256=hashlib.sha256((P/f'heldout_{i}.json').read_bytes()).hexdigest()))
out=dict(qualified=all(c['passed']for c in checks),checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],rows=rows,max_error=max(abs(c['error'])for c in checks),coefficient_sha256=hashlib.sha256((P/'COEFFICIENT_FREEZE.json').read_bytes()).hexdigest(),basis_retuned=False,scope='All8lockedNEWheldouts each4outputs,1%selected gate; not uniform domain accuracy.')
(P/'HELDOUT_ASSESSMENT.json').write_text(json.dumps(out,indent=2)+'\n');print(out['qualified'],out['max_error'],out['failed'])
