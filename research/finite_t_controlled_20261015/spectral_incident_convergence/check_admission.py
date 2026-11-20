"""Synthetic COPY admission probes; never mutate actual source/raw jobs."""
from pathlib import Path
import sys,json,copy,tempfile,shutil
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R));import postprocess as P
base=json.loads((R/'raw/EC_4_00.json').read_text());out=[]
variants={
 'incomplete':lambda d:d.update(complete=False),
 'missing node':lambda d:d['rows'].pop(),
 'wrong rule node':lambda d:d['rule']['nodes'].__setitem__(0,3.14),
 'wrong source':lambda d:d.update(source_pin_sha256='0'*64),
 'wrong writer':lambda d:d.update(worker_sha256='0'*64),
 'changed control':lambda d:d['controls'].update(phi_order=4),
 'changed bath':lambda d:d['bath'].update(mu_e=.18),
 'wrong current':lambda d:d['rows'][0]['result'].update(current_source='other'),
 'wrong units':lambda d:d['rows'][0]['result'].update(units='km^-1'),
 'falseempty both coefficients':lambda d:[d['rows'][0]['result'][k].update(value=0.,log=None,support='exact_kinematic_zero',reason='empty_support')for k in('j','true_chi')],
 'Boolean ordinary j':lambda d:d['rows'][0]['result']['j'].update(value=True,log=0.,reason='represented'),
 'nonfinite log chi':lambda d:d['rows'][0]['result']['true_chi'].update(log=float('nan')),
}
with tempfile.TemporaryDirectory(dir=R,prefix='owned_admission_')as directory:
 clone=Path(directory);(clone/'raw').mkdir()
 for name in('LOCK.json','SOURCE_PIN.json','nodes.py'):shutil.copy2(R/name,clone/name)
 originalR=P.R;P.R=clone
 try:
  (clone/'raw/EC_4_00.json').write_text(json.dumps(base));assert len(P.collect('EC',4)[0])==4
  for name,mutate in variants.items():
   d=copy.deepcopy(base);mutate(d);(clone/'raw/EC_4_00.json').write_text(json.dumps(d))
   try:P.collect('EC',4);accepted=True;error=None
   except ValueError as err:accepted=False;error=str(err)
   out.append(dict(name=name,passed=not accepted,error=error,synthetic_payload_JSON_text=json.dumps(d)))
 finally:P.R=originalR
(R/'admission_probes.json').write_text(json.dumps(dict(valid_copy_pass=True,checks=out,source_raw_unchanged=True),indent=2,allow_nan=False)+'\n');print('validcopy1PASS negative',sum(v['passed']for v in out),'PASS',sum(not v['passed']for v in out),'FAIL')
