"""Bounded serial exact-node evaluations; no interpolated spectral values."""
from pathlib import Path
import sys,json,hashlib,time,dataclasses,traceback
P=Path(__file__).resolve().parent;lock=json.loads((P/'LOCK.json').read_text());F=P/'frozen_beta_support';sys.path.insert(0,str(F))
from conditional_beta import evaluate,Rule
import conditional_beta
if Path(conditional_beta.__file__).resolve()!=(F/'conditional_beta.py').resolve():raise RuntimeError('Wrong explicit frozen source')
for name,h in lock['copied_sources'].items():
 if hashlib.sha256((P/name).read_bytes()).hexdigest()!=h:raise RuntimeError('Frozen source drift '+name)
start=int(sys.argv[1]);stop=int(sys.argv[2]);raw=P/'raw';raw.mkdir(exist_ok=True)
if not 0<=start<stop<=len(lock['jobs']) or stop-start>8:raise ValueError('Bounded <=8 exact jobs')
for job in lock['jobs'][start:stop]:
 path=raw/f'{job["id"]:03d}.json'
 if path.exists():raise RuntimeError('No physical raw overwrite')
 row={'job':job,'lock_sha256':hashlib.sha256((P/'LOCK.json').read_bytes()).hexdigest(),'candidate_sha256':lock['candidate_sha256'],'complete':False};begin=time.monotonic()
 try:
  row['result']=evaluate(job['E_MeV'],Rule(**job['kernel_rule']));row['complete']=True
 except Exception as e:row['error']={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
 row['seconds']=time.monotonic()-begin;path.write_text(json.dumps(row,indent=2)+'\n');print(job['id'],'PASS_RETURN' if row['complete'] else row['error'],flush=True)
 if not row['complete']:raise SystemExit(1)
