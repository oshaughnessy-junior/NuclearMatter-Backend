from pathlib import Path
import subprocess,sys
R=Path(__file__).resolve().parent
for ch in('EC','PC','NEUTRON_DECAY'):
 for rad in('empty','LTE'):
  cmd=[sys.executable,str(R/'public_agreement_successor.py'),ch,rad]
  r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=119)
  (R/'raw_successor'/f'public_agreement_{ch}_{rad}.stdout.txt').write_text(r.stdout);print(r.stdout,flush=True)
  if r.returncode:raise RuntimeError('Actual public agreement failed; preserve evidence and stop')
