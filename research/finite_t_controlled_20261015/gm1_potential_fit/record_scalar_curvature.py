"""Record already-solved exact scalar curvature; no refit or forward root rerun."""
from pathlib import Path
import sys,json
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P.parent/'rmf_reference'));sys.path.insert(0,str(P.parent/'rmf_phase'))
import gm1_finite_t as exact
from stability import implicit_hessian
r=json.loads((P/'results.json').read_text())
for row in r['rows']:
 H,d=implicit_hessian(exact.NuclearState(**row['exact']));D=d['canonical_scalar_curvature_fm_minus2'];row['exact_canonical_scalar_curvature_fm_minus2']=D
 name=f"{row['block']} reference canonical scalar curvature {(row['n'],row['T'],row['Y'])}"
 if not any(c['name']==name for c in r['checks']):r['checks'].append(dict(name=name,error=float(D<=0),tolerance=.1,passed=bool(D>0)))
r['passed']=all(c['passed'] for c in r['checks']);(P/'results.json').write_text(json.dumps(r,indent=2)+'\n');print(len(r['checks']),r['passed'],min(row['exact_canonical_scalar_curvature_fm_minus2'] for row in r['rows']))
