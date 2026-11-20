"""One targeted spectral16/tail8 followup; original failed gates unmodified."""
from pathlib import Path
import json,time,math,copy
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import expit
import box as B
HERE=Path(__file__).resolve().parent;old=json.loads((HERE/'results.json').read_text());sens=json.loads((HERE/'sensitivity_results.json').read_text());checks=[];cases=[];start=time.monotonic()
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
def observe(g,Tr):
 E=np.array(g['E']);r=B.evolve(g,Tr,-.7,steps=64);return dict(Tr=Tr,run=r,initial=B.collision(g,expit((-.7-E)/Tr)))
g16=B.grid(16,24.,angle=24,inner=24,tail=60.)
for Tr in[1.3,3.]:
 a=next(x['fine']for x in sens['cases']if x['label']=='spectral8_12_at_upper24'and x['Tr']==Tr);b=observe(g16,Tr)
 gate(str(Tr)+' new spectral12to16 same onepercent heat gate',a['final_heat']/b['run']['matter_heat']-1,.01);cases.append(dict(label='spectral12_16',Tr=Tr,grid=g16,coarse=a,fine=b))
# Tail8: preserve all previous8 core nodes and original physical kernel entries.
base=old['grids'][-1];E=list(base['E']);w=list(base['w']);x,v=leggauss(8)
for lo,hi in[(24.,36.),(36.,48.)]:E.extend(((x+1)*(hi-lo)/2+lo).tolist());w.extend((v*(hi-lo)/2).tolist())
E=np.array(E);w=np.array(w);W=w*E**2/(2*math.pi**2*B.HC**3);K=np.zeros((24,24));K[:8,:8]=base['K'];st=time.monotonic()
for i in range(24):
 for j in range(24):
  if i!=j and not(i<8 and j<8):
   if time.monotonic()-st>90:raise TimeoutError('Targeted tail experiment bounded')
   K[i,j]=B.edge(float(E[i]),float(E[j]),angle=24,inner=24,tail=60.)
g=dict(E=E.tolist(),w=w.tolist(),W=W.tolist(),K=K.tolist(),A=(W[:,None]*w[None,:]*K).tolist(),N=24,upper=48.,tail_panel_order=8)
for Tr in[1.3,3.]:
 a=next(x for x in sens['cases']if x['label']=='upper_extension'and x['upper']==48. and x['tail_order']==6 and x['Tr']==Tr);b=observe(g,Tr)
 for name,val in [('initial_heat',b['initial']['matter_energy']),('final_heat',b['run']['matter_heat'])]:gate(str(Tr)+' isolated new tail6to8 '+name,a['observables'][name]/val-1,3e-5)
 cases.append(dict(label='tail6_8',Tr=Tr,coarse=a['observables'],fine=b,grid=g))
# Actual FD evolution, not only its instantaneous stationary residual.
g0=base;fd=B.evolve(g0,B.T,-.7,steps=64);gate('actual FD stationary evolution',max(abs(np.array(fd['final'])-np.array(fd['initial']))),3e-13)
# Structured controls fail before unknown state can contaminate a ledger.
for label,mutate in [('NaN weight',lambda g:g['W'].__setitem__(0,float('nan'))),('negative edge',lambda g:g['A'][0].__setitem__(1,-1)),('diagonal self flux',lambda g:g['A'][0].__setitem__(0,1))]:
 z=copy.deepcopy(base);mutate(z)
 try:B.collision(z,fd['initial']);bad=1
 except ValueError:bad=0
 gate('reject '+label,bad,0)
result=dict(checks=checks,passed=sum(x['passed']for x in checks),failed=[x for x in checks if not x['passed']],cases=cases,FD_evolution=fd,pins=B.SOURCE_PINS,seconds=time.monotonic()-start)
(HERE/'followup_results.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k]for k in ['passed','failed','seconds']},indent=2))
