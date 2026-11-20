from pathlib import Path
import sys,json,traceback,dataclasses,numpy as np,time
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P))
from pr55_reference import weak_reference as W
from pr55_reference import _weak_current as C
b=W.FreeElectronBath(2.,919.8287578100662,916.8449303084382,.17446333200826397)
c=dataclasses.replace(W.SpectralControls.fine(),budget_seconds=85.)
z={'bath':dataclasses.asdict(b),'controls':dataclasses.asdict(c),'energy_MeV':.1,'channel':'PC','complete':False}
t=time.monotonic()
try:
 row=W.spectral_coefficients('PC',b,.1,c);z['result']=dataclasses.asdict(row);z['complete']=True
except Exception as e:
 z['error']={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
 tb=e.__traceback__
 while tb:
  if tb.tb_frame.f_code.co_name=='differential':
   v=tb.tb_frame.f_locals; z['witness']={k:float(v[k]) for k in ('energy','el','cosine','ma','mb','omega','q')};z['witness']['intervals']=v['intervals']
   for k in ('ea','ca','b'):
    if k in v:z['witness'][k+'_minmax']=[float(np.min(v[k])),float(np.max(v[k]))]
   if 'b' in v:
    err=C.dot(v['b'],v['b'])/v['mb']**2-1;idx=int(np.argmax(abs(err)));z['witness']['worst_index']=idx;z['witness']['worst_shell_relative']=float(err[idx]);z['witness']['worst_initial']=v['a'][idx].tolist();z['witness']['worst_final']=v['b'][idx].tolist()
  tb=tb.tb_next
z['seconds']=time.monotonic()-t;(P/'ORIGINAL_FAILURE.json').write_text(json.dumps(z,indent=2)+'\n');print(json.dumps(z,indent=2))
