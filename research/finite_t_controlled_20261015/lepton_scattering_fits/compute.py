"""Compute preregistered actual source references with per-state checkpoints."""
from pathlib import Path
import sys,json,time,math,importlib.util
P=Path(__file__).parent;sys.dont_write_bytecode=True
sp=importlib.util.spec_from_file_location('moment_fit_source_owned',P/'reference.py');R=importlib.util.module_from_spec(sp);sys.modules[sp.name]=R;sp.loader.exec_module(R)

def main(part):
 reg=json.loads((P/'PREREGISTRATION.json').read_text());assert part in ['training','heldout'];lock=P/'FITTING_LOCK.json'
 if part=='heldout'and not lock.is_file():raise ValueError('Model must be locked before any heldout references')
 rows=[];checks=[];start=time.monotonic();dest=P/(part+'_references.json')
 if dest.exists():raise RuntimeError('Refusing to overwrite frozen reference evidence; preserve/copy into a fresh owned leaf first')
 for E in reg[part+'_MeV']:
  a=R.detailed(E,R.COARSE);b=R.detailed(E,R.FINE)
  for key in reg['positive_observables']:
   err=a['total'][key]/b['total'][key]-1;checks.append(dict(name='quadrature '+str((E,key)),error=err,tolerance=1e-4,passed=abs(err)<=1e-4))
  err=(a['total']['neutrino_energy_change']-b['total']['neutrino_energy_change'])/(b['total']['rate']*R.T);checks.append(dict(name='quadrature signedRT '+str(E),error=err,tolerance=1e-4,passed=abs(err)<=1e-4))
  for target in b['targets']:
   checks.append(dict(name='energy eventledger '+str((E,target['positron'])),error=target['energy_ledger']/target['outgoing_energy_moment'],tolerance=3e-12,passed=abs(target['energy_ledger']/target['outgoing_energy_moment'])<=3e-12))
  rows.append(dict(E=E,coarse=a,fine=b));out=dict(part=part,rows=rows,checks=checks,passed=sum(z['passed']for z in checks),failed=[z for z in checks if not z['passed']],seconds=time.monotonic()-start);dest.write_text(json.dumps(out,indent=2));print(json.dumps(dict(part=part,E=E,passed=out['passed'],failed=out['failed'])),flush=True)
if __name__=='__main__':main(sys.argv[1])
