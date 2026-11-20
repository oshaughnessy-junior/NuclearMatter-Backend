"""Reconstruct coefficients from training data alone and audit test separation."""
from pathlib import Path
import json,math
import numpy as np
P=Path(__file__).resolve().parent;root=P.parent/'gm1_potential_fit';final=json.loads((root/'results.json').read_text());old=json.loads((root/'initial_Y_polynomial/results.json').read_text());coeff=json.loads((root/'coefficients.json').read_text())['fields'];checks=[]
def rec(x,n):
 v=[1.,x]
 for k in range(2,n):v.append(2*x*v[-1]-v[-2])
 return v[:n]
V=[]
for row in final['training']:
 x=(row['n']-.22)/.04;z=(2*(1-2*row['Y'])**2-(.34**2+.64**2))/(.64**2-.34**2)
 V.append(np.outer(rec(x,5),rec(z,4)).ravel())
V=np.array(V);differences={}
for field in ['cold','T2','T4']:
 computed=np.linalg.solve(V,np.array([r[field+'_field']for r in final['training']])).reshape(5,4);error=float(np.max(abs(computed-np.array(coeff[field]))));differences[field]=error;checks.append(dict(name='coefficients reproduced from training only '+field,error=error,tolerance=1e-9,passed=error<1e-9))
newset={(r['n'],r['T'],r['Y'])for r in final['rows']};oldset={(r['n'],r['T'],r['Y'])for r in old['rows']};overlap=sorted(newset&oldset);checks.append(dict(name='new final holdouts disjoint old diagnostic coordinates',passed=not overlap))
train={(r['n'],T,r['Y'])for r in final['training']for T in [0.,2.]};checks.append(dict(name='final holdouts disjoint fit states',passed=not(newset&train)))
result=dict(passed=all(x['passed']for x in checks),checks=checks,max_coefficient_differences=differences,old_new_exact_tuple_overlap=overlap,condition_number=float(np.linalg.cond(V)),limits='Numerical coefficients depend only on declared training targets. Model-selection history/predeclaration cannot be proven from data disjointness; author explicitly disclosed earlier holdout-informed basis selection.')
(P/'data_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
