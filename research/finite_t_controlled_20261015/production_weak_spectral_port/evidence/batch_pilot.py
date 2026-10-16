from pathlib import Path
import sys,subprocess
R=Path(__file__).resolve().parents[1]
for ch in ('EC','PC','NEUTRON_DECAY'):
 for order in (2,4):
  for name in ('empty','LTE','noneq'):
   cmd=[sys.executable,str(R/'evidence/range_pilot.py'),ch,str(order),name]
   r=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=119)
   (R/'evidence'/f'pilot_{ch}_{order}_{name}.stdout.txt').write_text(r.stdout)
   print(r.stdout,flush=True)
