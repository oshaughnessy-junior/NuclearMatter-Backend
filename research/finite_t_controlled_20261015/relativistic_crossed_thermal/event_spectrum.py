"""Whole selected spectrum with independently Q-rest integrated spinor kernel."""
from pathlib import Path
import sys,json,time,math
import numpy as np
from scipy.special import logsumexp
sys.dont_write_bytecode=True
from kernel import *
from oracles import event_reference
H=Path(__file__).parent;start=time.monotonic();bath,m=matched_npe(.03,1e-12,.3)
angle_order=8;lepton_order=12;Enu=.3;widths=12.
x,w=leggauss(angle_order);y,v=leggauss(lepton_order);logs=[[],[]];count=0
for c,wc in zip(x,w):
 def shell(E):
  p=math.sqrt(max(0.,(E-ME)*(E+ME)));return E+Enu+math.sqrt(MP*MP+p*p+Enu*Enu+2*p*Enu*c)-MN
 center=brentq(shell,ME,MN-MP,xtol=2e-13);p=math.sqrt((center-ME)*(center+ME));q=math.sqrt(Enu*Enu+p*p+2*Enu*p*c);width=q*math.sqrt(bath.T/MN)
 edges=sorted({max(ME,center-widths*width),center,max(ME,center-3*width),center+3*width,center+widths*width})
 for lo,hi in zip(edges[:-1],edges[1:]):
  for xx,ww in zip(y,v):
   if time.monotonic()-start>105:raise TimeoutError('Independent event spectrum105s budget; no acceptance')
   E=lo+(xx+1)*(hi-lo)/2;weight=wc*ww*(hi-lo)/2
   values=event_reference(bath,Enu,E,float(c),order=20)
   for i in range(2):
    if values[i]is not None:logs[i].append(math.log(weight)+values[i])
   count+=1
reference=[float(logsumexp(a))for a in logs]
matched=spectral(bath,.3,angle_order=angle_order,lepton_order=lepton_order,energy_order=16,widths=widths)
refined=json.loads((H/'results.json').read_text());fine=[r['spectrum']for r in refined['rows']if'spectrum'in r][-1]
checks=[]
for i,key in enumerate(['j','chi']):
 for label,r in [('sameouter',matched),('refinedouter',fine)]:
  err=math.expm1(r[key]['log']-reference[i]);checks.append(dict(name=label+' '+key,error=err,tolerance=3e-7,passed=abs(err)<3e-7))
out=dict(passed=all(c['passed']for c in checks),checks=checks,event_reference=dict(logj=reference[0],logchi=reference[1],evaluations=count,controls=dict(angle=angle_order,lepton=lepton_order,Qrest_order=20,phi=8,widths=widths)),same_outer_tensor=matched,runtime_seconds=time.monotonic()-start)
(H/'event_spectrum_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(passed=out['passed'],checks=checks,runtime=out['runtime_seconds'])))
