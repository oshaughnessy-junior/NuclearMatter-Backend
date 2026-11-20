from pathlib import Path
import sys,json,subprocess,time
R=Path(__file__).resolve().parent;lock=json.loads((R/'LOCK.json').read_text());ch=sys.argv[1]
if ch not in lock['channels']:raise ValueError('Named locked channel required')
start=time.monotonic()
for order in lock['incident_orders']:
 for i in range(0,order,lock['resources']['chunk_nodes']):
  p=R/'raw'/f'{ch}_{order}_{i:02d}.json'
  if p.exists():
   data=json.loads(p.read_text())
   if not data.get('complete'):raise RuntimeError('Incomplete previous chunk requires separate preserved repair: '+str(p))
   continue
  command=[sys.executable,str(R/'nodes.py'),ch,str(order),str(i)]
  try:r=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=119)
  except subprocess.TimeoutExpired as err:
   (R/'raw'/f'{ch}_{order}_{i:02d}.TIMEOUT.txt').write_text(str(err));raise
  (R/'raw'/f'{ch}_{order}_{i:02d}.stdout.txt').write_text(r.stdout);print(r.stdout,flush=True)
  if r.returncode:raise RuntimeError('Actual job failed; no complete result admitted')
print(ch,'ALLDONE',time.monotonic()-start,flush=True)
