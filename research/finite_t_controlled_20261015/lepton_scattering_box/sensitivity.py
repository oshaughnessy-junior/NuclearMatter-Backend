"""Separate spectral, time, internal-edge and physical upper-bound effects."""
from __future__ import annotations
from pathlib import Path
import json,time,math
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import expit
import box as B
HERE=Path(__file__).resolve().parent;start=time.monotonic();old=json.loads((HERE/'results.json').read_text());checks=[];cases=[]
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
def dump():(HERE/'sensitivity_results.json').write_text(json.dumps(dict(checks=checks,cases=cases,seconds=time.monotonic()-start),indent=2))
def observe(g,Tr):
 E=np.array(g['E']);f=expit((-.7-E)/Tr);c=B.collision(g,f);r=B.evolve(g,Tr,-.7,steps=64)
 return dict(Tr=Tr,initial_heat=c['matter_energy'],activity=c['activity'],final_heat=r['matter_heat'],final_radiation_energy=r['radiation_energy_final'],run=r)
base=old['grids'][-1]
# Fine-grid time error already below spatial differences; preserve explicit
# failed six/eight spectral gates instead of calling all conservation passes accuracy.
for Tr in [1.3,3.]:
 heats=[next(x['run']['matter_heat']for x in old['evolutions']if x['N']==N and x['run']['Tr']==Tr and x['run']['steps']==32)for N in[4,6,8]]
 gate(str(Tr)+' spectral6to8 onepercent gate',heats[1]/heats[2]-1,.01)
 cases.append(dict(label='spectral4_6_8_at_upper24',Tr=Tr,heats=heats))
refined=B.grid(8,24.,angle=40,inner=32,tail=80.)
for Tr in[1.3,3.]:
 a,b=observe(base,Tr),observe(refined,Tr)
 for key in['initial_heat','final_heat','activity']:gate(str(Tr)+' full8edge internal refinement '+key,a[key]/b[key]-1,3e-6)
 cases.append(dict(label='internal_kernel_refinement',Tr=Tr,base=a,fine=b))
dump()
# Additional spectral grid: physical edges are calculated, never interpolated.
g12=B.grid(12,24.,angle=24,inner=24,tail=60.)
for Tr in[1.3,3.]:
 a,b=observe(base,Tr),observe(g12,Tr);gate(str(Tr)+' spectral8to12 onepercent gate',a['final_heat']/b['final_heat']-1,.01)
 cases.append(dict(label='spectral8_12_at_upper24',Tr=Tr,coarse=a,fine=b))
dump()

def extend(core,upper,tail_order):
 # Core nodes/weights/edges held BITWISE fixed. Separate tail-panel nodes.
 E=list(core['E']);w=list(core['w']);x,v=leggauss(tail_order)
 for lo,hi in [(24.,36.),(36.,48.)]:
  if hi<=upper:E.extend(((x+1)*(hi-lo)/2+lo).tolist());w.extend((v*(hi-lo)/2).tolist())
 E=np.array(E);w=np.array(w);W=w*E**2/(2*math.pi**2*B.HC**3);K=np.zeros((len(E),len(E)));N=core['N'];K[:N,:N]=core['K'];st=time.monotonic()
 for i in range(len(E)):
  for j in range(len(E)):
   if i!=j and not(i<N and j<N):
    if time.monotonic()-st>90:raise TimeoutError('Tail extension experiment budget')
    K[i,j]=B.edge(float(E[i]),float(E[j]),angle=24,inner=24,tail=60.)
 A=W[:,None]*w[None,:]*K
 return dict(E=E.tolist(),w=w.tolist(),W=W.tolist(),K=K.tolist(),A=A.tolist(),N=len(E),upper=upper,tail_panel_order=tail_order,core_bitwise_unchanged=bool(np.array_equal(K[:N,:N],np.array(core['K']))),seconds=time.monotonic()-st)
for upper,tailorder in[(36.,4),(48.,4),(48.,6)]:
 g=extend(base,upper,tailorder);gate(str((upper,tailorder))+' frozen core for independent physical cutoff',int(not g['core_bitwise_unchanged']),0)
 for Tr in[1.3,3.]:
  o=observe(g,Tr);E=np.array(g['E']);W=np.array(g['W']);w=np.array(g['w']);K=np.array(g['K']);f=expit((-.7-E)/Tr);F=np.array(g['A'])*f[:,None]*(1-f[None,:]);N=8
  outgoing=float(np.sum(F[:N,N:]));incoming=float(np.sum(F[N:,:N]));net=incoming-outgoing
  cases.append(dict(label='upper_extension',upper=upper,tail_order=tailorder,Tr=Tr,observables=o,cross_boundary_forward=outgoing,cross_boundary_backward=incoming,core_number_source=net,grid=g))
 dump()
# Compare same core and same tail interval with a NEW tail quadrature.
for Tr in[1.3,3.]:
 a=next(x for x in cases if x['label']=='upper_extension'and x['upper']==48. and x['tail_order']==4 and x['Tr']==Tr);b=next(x for x in cases if x['label']=='upper_extension'and x['upper']==48. and x['tail_order']==6 and x['Tr']==Tr)
 for key in['initial_heat','final_heat','activity']:gate(str(Tr)+' isolated tail quadrature4to6 '+key,a['observables'][key]/b['observables'][key]-1,3e-5)
dump();print(json.dumps(dict(passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],seconds=time.monotonic()-start),indent=2))
