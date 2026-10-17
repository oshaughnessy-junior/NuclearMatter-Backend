from pathlib import Path
import sys,json,time,dataclasses,hashlib,traceback
P=Path(__file__).resolve().parent
mode=sys.argv[1];E=float(sys.argv[2]);tag=sys.argv[3];out=P/(tag+'.json')
if out.exists():raise RuntimeError('Refuse raw-result overwrite')
t=time.monotonic();row={'mode':mode,'E':E,'tag':tag,'complete':False}
try:
 if mode=='old':
  from beta_pr55_reference import weak_reference as W
  c=dataclasses.replace(W.SpectralControls.fine(),tail=float(sys.argv[4]),budget_seconds=85.)
  b=W.FreeElectronBath(2.,919.8287578100662,916.8449303084382,.17446333200826397)
  r=W.spectral_coefficients('NEUTRON_DECAY',b,E,c);row['result']=dataclasses.asdict(r)
 else:
  from conditional_beta import evaluate,Rule
  profiles={'panel':Rule(12,16,16,10,'panels'),'semi16':Rule(16,20,20,14,'laguerre'),'semi24':Rule(24,24,24,16,'laguerre')}
  row['result']=evaluate(E,profiles[sys.argv[4]])
 row['complete']=True
except Exception as e:row['error']={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
row['seconds']=time.monotonic()-t;row['candidate_sha256']=hashlib.sha256((P/'conditional_beta.py').read_bytes()).hexdigest();out.write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row,indent=2))
