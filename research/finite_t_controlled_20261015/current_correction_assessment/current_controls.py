"""Direct source Dirac traces and exact component switch/reciprocity controls."""
from pathlib import Path
import sys,importlib.util,json,hashlib,math
import numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parent;sys.dont_write_bytecode=True

def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
C=load('F2_control_frozen_trace_current',ROOT/'muon_full_current/current.py');K=load('F2_control_owned_scalar_functions',P/'configured_currents.py');EC=load('F2_control_original_EC',ROOT/'hot_electron_capture_moments/electron_capture.py');PC=load('F2_control_original_PC',ROOT/'review_pc_moments/reference.py');checks=[];rows=[]
def ck(n,e,t=3e-10):checks.append(dict(name=n,error=float(e),gate=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
for anti in [False,True]:
 ch=C.V.Channel(K.MP if anti else K.MN,K.MN if anti else K.MP,K.ME,antineutrino=anti)
 for en,c in [(4.7,-.61),(17.3,.22),(48.9,.78)]:
  p=C.V.momenta(ch,en,c);q=p['pb']-p['pa'];t=C.V.dot(q,q);full=C.FormFactors().values(t,K.MW);off=C.FormFactors(weak_magnetism=False).values(t,K.MW);args=[p[k][None,:]for k in['k','pa','l','pb']];fn=K.pc_full if anti else K.ec_full;fo=K.pc_F2zero if anti else K.ec_F2zero;original=PC.amplitude if anti else EC.scalar_amplitude;ck('configured baseline original scalar '+str((anti,en)),float(fn(*args)[0]/original(*args)[0]-1),3e-14)
  for mode,ff,f in [('baseline',full,fn),('F2zero',off,fo)]:
   value=float(f(*args)[0]);trace=C.trace(ch,p,ff)/2;ck('avg4 scalar vs direct Dirac '+str((anti,en,mode)),value/trace-1);ck('positive point amplitude '+str((anti,en,mode)),0 if value>0 else 1,0);rows.append(dict(anti=anti,Enu=en,cos=c,variant=mode,scalar=value,trace=trace,form_factors=ff))
  ck('only F2 coefficient changed '+str((anti,en)),max(abs(full[i]-off[i])for i in[0,1,3]),0);ck('F2 exactly0 '+str((anti,en)),off[2],0)
  vf=C.vertices(q,K.MW,full);vo=C.vertices(q,K.MW,off);mag=np.array(vf['F']);qcontract=sum(C.V.METRIC[u]*q[u]*mag[u]for u in range(4));scale=max(1.,float(np.max(abs(mag))))*max(1.,float(max(abs(q))));ck('magnetic current transverse '+str((anti,en)),float(np.max(abs(qcontract)))/scale,3e-13);ck('off magnetic current exactly zero '+str((anti,en)),float(np.max(abs(np.array(vo['F'])))),0)
  for term in['V','A','P']:ck('other vertex component unchanged '+str((anti,en,term)),float(np.max(abs(np.array(vf[term])-np.array(vo[term])))),0)
  for mode,ff in [('baseline',full),('F2zero',off)]:
   v=C.vertices(q,K.MW,ff);minus=C.vertices(-q,K.MW,ff);total=[sum(v[k][u]for k in v)for u in range(4)];reverse=[sum(minus[k][u]for k in minus)for u in range(4)];er=max(np.max(abs(C.V.adjoint(total[u])-reverse[u]))for u in range(4));ck('Dirac adjoint reversed q '+str((anti,en,mode)),float(er),3e-13)
out=dict(checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],rows=rows);(P/'current_controls.json').write_text(json.dumps(out,indent=2)+'\n');print(out['passed'],out['failed'])
