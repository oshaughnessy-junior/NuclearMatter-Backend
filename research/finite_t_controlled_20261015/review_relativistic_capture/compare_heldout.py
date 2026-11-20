"""Only completed heldout physical runs enter comparisons."""
from pathlib import Path
import json,math,hashlib
P=Path(__file__).resolve().parent;ROOT=P.parent
runs={k:json.loads((P/(k+'.json')).read_text())for k in ('capture12','capture16','events10','events14')};checks=[];rows=[]
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and error<tol)))
base=runs['capture16'];matter=base['matter'];target=matter['neutron']['density'];expected=-(157.-(matter['muon']['chemical_potential']+matter['proton']['chemical_potential']-matter['neutron']['chemical_potential']))/matter['temperature']
for name,record in runs.items():
 gate(name+' exact fullstate',float(record['matter']!=matter),.1)
 gate(name+' exact current immutable source data',float(any(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()!=h for f,h in record['source_hashes_before'].items())),.1)
 gate(name+' beforeafter hashes same',float(record['source_hashes_before']!=record['source_hashes_after']),.1)
 r=record['result'];gate(name+' independent inverse KMS',abs(r['j']['log']-r['chi']['log']-expected),3e-10)
 if name.startswith('events'):
  gate(name+' actual initial FDdensity',abs(r['density']/target-1),3e-9);gate(name+' actualfourmomentum residual',r['max_fourmomentum_residual'],2e-14)
 rows.append(dict(mode=name,j=r['j'],chi=r['chi'],runtime_seconds=record['runtime_seconds']))
for name1,name2 in [('capture12','capture16'),('events10','events14'),('capture16','events14')]:
 for key in ('j','chi'):gate(name1+' versus '+name2+' '+key,abs(math.expm1(runs[name1]['result'][key]['log']-runs[name2]['result'][key]['log'])),3e-6)
(P/'heldout_comparison.json').write_text(json.dumps(dict(passed=all(c['passed']for c in checks),checks=checks,rows=rows,state=matter,selected_tolerance=3e-6,scope='One additional neutrino-muon capture state and energy only; not a uniform pilot API certificate'),indent=2)+'\n');print(sum(c['passed']for c in checks),'/',len(checks))
for c in checks:
 if not c['passed']:print('FAIL',c)
