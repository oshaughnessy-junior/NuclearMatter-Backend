from pathlib import Path
import json,hashlib,time
from conditional_pc import evaluate,Rule
P=Path(__file__).resolve().parent
rows=[]
for E in (.1,.15):
 for tag,r in [('coarse',Rule(8,10,10,4)),('inner',Rule(12,16,16,6)),('higher',Rule(16,22,22,8)),('tail90',Rule(16,22,22,8,90.)),('literal',Rule(12,16,16,6,70.,True))]:
  try:z=evaluate(E,r);z['tag']=tag;rows.append(z);print(E,tag,z['log_j_s-1'],z['log_true_chi_s-1'],z['seconds'],flush=True)
  except Exception as e:rows.append({'E':E,'tag':tag,'error':str(e)});print('ERROR',E,tag,str(e),flush=True)
  (P/'CONVERGENCE_RAW.json').write_text(json.dumps({'candidate_sha256':hashlib.sha256((P/'conditional_pc.py').read_bytes()).hexdigest(),'rows':rows},indent=2)+'\n')
