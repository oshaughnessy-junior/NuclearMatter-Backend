"""Bounded physical edge qualification and closed finite-grid scattering study."""
from __future__ import annotations
import json,time,math
from pathlib import Path
import numpy as np
from scipy.special import expit
import box as B
import reference_edge as R
HERE=Path(__file__).resolve().parent;checks=[];start=time.monotonic()
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
def dump(name,x):(HERE/name).write_text(json.dumps(x,indent=2,default=lambda y:y.item()if isinstance(y,np.generic)else str(y)))
def main():
 material=B.bath();gate('independent actual pair bath forward',material['residual'],2e-9);edge_rows=[]
 for E,F in [(3.2,8.9),(14.1,10.7),(1.1,2.4)]:
  a=B.edge(E,F,angle=16,inner=20);b=B.edge(E,F,angle=32,inner=32,tail=80.);ref=R.edge(B.T,B.MU,E,F,B.COUPLING.cv,B.COUPLING.ca);reverse=B.edge(F,E,angle=32,inner=32,tail=80.)
  gate('selected absolute independent r/q-shell edge '+str((E,F)),b/ref-1,3e-7);gate('selected edge refinement '+str((E,F)),a/b-1,3e-6);gate('selected reciprocal with DOS '+str((E,F)),reverse/b/((E/F)**2*math.exp((F-E)/B.T))-1,2e-10);edge_rows.append(dict(E=E,F=F,coarse=a,fine=b,independent=ref,reverse=reverse))
 dump('edge_qualification.json',dict(material=material,edges=edge_rows,checks=checks,pins=B.SOURCE_PINS))
 grids=[];evolutions=[];initials=[]
 for N in [4,6,8]:
  g=B.grid(N,24.,angle=24,inner=24,tail=60.);grids.append(g);gate(str(N)+' physical edge KMS',g['kms_log_residual'],3e-10)
  E=np.array(g['E']);W=np.array(g['W']);lte=B.collision(g,expit((-.7-E)/B.T));gate(str(N)+' FD actual-kernel stationarity',max(abs(np.array(lte['dn'])))/lte['activity'],3e-12)
  for Tr in [1.3,3.]:
   init=B.collision(g,expit((-.7-E)/Tr));initials.append(dict(N=N,Tr=Tr,collision=init));scale=init['activity'];gate(str((N,Tr))+' independent row vs paired occupation derivative',max(abs(np.array(init['dn'])-np.array(init['independent_dn'])))/scale,3e-13);gate(str((N,Tr))+' entropy micro/macro',abs(init['entropy_production']-init['macro_entropy'])/scale,3e-11)
   # hotter radiation should heat bath; colder radiation should cool bath.
   gate(str((N,Tr))+' physical heat direction',int((init['matter_energy']>0)!=(Tr>B.T)),0)
   for steps in ([16,32,64]if N==8 else[32]):
    run=B.evolve(g,Tr,-.7,duration=.08,steps=steps);evolutions.append(dict(N=N,run=run));energy_exchange=abs(run['matter_heat']);gate(str((N,Tr,steps))+' exact closed number',abs(run['number_ledger'])/run['number_initial'],3e-13);gate(str((N,Tr,steps))+' energy and thermostat ledger',abs(run['energy_ledger'])/energy_exchange,3e-12);gate(str((N,Tr,steps))+' total radiation plus thermostat entropy',min(0.,run['entropy_radiation']+run['entropy_bath'])/run['number_initial'],1e-14)
  dump('partial.json',dict(material=material,grids=grids,evolutions=evolutions,initials=initials,checks=checks,pins=B.SOURCE_PINS))
 # Time-convergence uses weighted conserved occupation rather than a scalar sign.
 time_errors=[]
 for Tr in [1.3,3.]:
  rr=[x['run']for x in evolutions if x['N']==8 and x['run']['Tr']==Tr];W=np.array(grids[-1]['W']);norm=lambda a,b:float(W@abs(np.array(a['final'])-np.array(b['final'])))/rr[-1]['number_initial'];d1=norm(rr[0],rr[1]);d2=norm(rr[1],rr[2]);p=math.log2(d1/d2);gate(str(Tr)+' observed RK2 order',p-2,.25);time_errors.append(dict(Tr=Tr,difference16_32=d1,difference32_64=d2,order=p))
 result=dict(material=material,edges=edge_rows,grids=grids,evolutions=evolutions,initial_collisions=initials,time_errors=time_errors,checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],pins=B.SOURCE_PINS,seconds=time.monotonic()-start,scope='closed finite isotropic energy box, fixed actual pair bath, nue only; no continuum/material feedback claim')
 dump('results.json',result);print(json.dumps({k:result[k]for k in ['passed','failed','seconds','time_errors']},indent=2))
if __name__=='__main__':main()
