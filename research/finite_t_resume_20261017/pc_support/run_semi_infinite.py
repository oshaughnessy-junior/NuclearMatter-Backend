from pathlib import Path
import json,hashlib
from conditional_pc import Rule,evaluate
P=Path(__file__).resolve().parent
rows=[]
for E in (.1,.15):
 for n in (8,16,24,32):
  r=evaluate(E,Rule(radial=n,incoming=22,outgoing=22,phi=8,radial_family='laguerre'));rows.append(r)
  print(E,n,r['log_j_s-1'],r['log_true_chi_s-1'],flush=True)
  (P/'SEMI_INFINITE_RADIAL_RAW.json').write_text(json.dumps({'candidate_sha256':hashlib.sha256((P/'conditional_pc.py').read_bytes()).hexdigest(),'rows':rows},indent=2)+'\n')
