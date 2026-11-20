"""Separate energy, azimuth and actual thermal-tail refinements."""
from adapter import *
import json,time
P=Path(__file__).resolve().parent;d=json.loads((P/'results.json').read_text());checks=[];rows=[];start=time.monotonic()
def arr(v):return np.array(v['real'])+1j*np.array(v['imag'])
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and error<tol)))
for row in d['rows']:
 s=OrderedState(**row['state']);ref=row['runs'][-1];r0=ref['inputs'];rec=dict(label=row['label'],runs=[]);rows.append(rec)
 for label,kw in [('energy_only',dict(energy_order=24,phi_order=16,tail=80.)),('phi_only',dict(energy_order=40,phi_order=8,tail=80.)),('tail_only',dict(energy_order=40,phi_order=16,tail=50.))]:
  r=differential(s,**r0,**kw)
  for key in ('j','chi'):gate(row['label']+label+key,abs(math.expm1(r[key]['log']-ref[key]['log'])),3e-7)
  for branch in ('absorption','emission'):gate(row['label']+label+branch,np.max(abs(r['response'][branch]['tensor']-arr(ref['response'][branch]['tensor'])))/np.max(abs(arr(ref['response'][branch]['tensor']))),3e-7)
  rec['runs'].append(dict(control=label,j=r['j'],chi=r['chi'],bounds=r['response']['panels'],controls=r['response']['controls']))
(P/'refinement_results.json').write_text(json.dumps(dict(passed=all(c['passed']for c in checks),checks=checks,rows=rows,runtime_seconds=time.monotonic()-start),indent=2)+'\n');print(sum(c['passed']for c in checks),'/',len(checks),'runtime',time.monotonic()-start)
