"""Preregistered development references; no fresh validation evaluation."""
from pathlib import Path
import importlib.util,sys,json,hashlib,time
P=Path(__file__).parent;OLD=P.parent/'lepton_scattering_fits'
spec=importlib.util.spec_from_file_location('joint_fit_reference_owned',P/'reference.py');R=importlib.util.module_from_spec(spec);sys.modules[spec.name]=R;spec.loader.exec_module(R)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 if (P/'training.json').exists()or(P/'FITTING_LOCK.json').exists():raise RuntimeError('Frozen data exists')
 reg=json.loads((P/'PREREGISTRATION.json').read_text());rows={};origins={}
 for name in['training_references.json','heldout_references.json']:
  data=json.loads((OLD/name).read_text())
  if data['failed']:raise ValueError('Unqualified historical reference')
  for z in data['rows']:rows[z['E']]=z;origins[str(z['E'])]=dict(path=str(OLD/name),sha256=sha(OLD/name),designation='KNOWN DEVELOPMENT, never v2 heldout')
 for E in[8.14,8.19,23.17]:
  a=R.detailed(E,R.COARSE);b=R.detailed(E,R.FINE);z=dict(E=E,coarse=a,fine=b);rows[E]=z;origins[str(E)]={'designation':'fresh recomputed known development'}
  (P/'development_partial.json').write_text(json.dumps(dict(rows=list(rows.values()),origins=origins),indent=2))
 checks=[]
 for E in reg['training_MeV']:
  z=rows[E];a=z['coarse']['total'];b=z['fine']['total']
  for k in['rate','transport_rate','outgoing_energy_moment']:checks.append(dict(E=E,name=k,error=a[k]/b[k]-1,gate=1e-4,passed=abs(a[k]/b[k]-1)<=1e-4))
  er=(a['neutrino_energy_change']-b['neutrino_energy_change'])/(2*b['rate']);checks.append(dict(E=E,name='signed_RT',error=er,gate=1e-4,passed=abs(er)<=1e-4))
  if z['fine']['pins']!=R.PINS:raise ValueError('Mixed source identity')
 out=dict(rows=[rows[e]for e in reg['training_MeV']],origins=origins,checks=checks,failed=[z for z in checks if not z['passed']],preregistration_sha=sha(P/'PREREGISTRATION.json'))
 (P/'training.json').write_text(json.dumps(out,indent=2));print(json.dumps(dict(count=len(out['rows']),failed=out['failed']),indent=2))
if __name__=='__main__':main()
