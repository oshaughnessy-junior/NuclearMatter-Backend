"""Constrained finite-grid LTE perturbations independent of box.step/rates."""
from pathlib import Path
import sys,json,math,time,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent.parent;HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'thermal_box'))
import box as B
start=time.monotonic();raw=json.loads((ROOT/'thermal_box/results.json').read_text());eq=raw['LTE'];n=raw['nB'];g=B.grid(raw['grid']['edges']);f=np.asarray(eq['f']);T,Y=eq['T'],eq['Ye'];matter=B.eos(T,n,Y);initial=B.invariants(T,Y,f,n,g);checks=[];samples=[]
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and abs(error)<=tol)))
for key in ('energy_MeV_per_baryon','electron_lepton_per_baryon'):
 gate('reconstructed LTE '+key,initial[key]-raw['initial']['totals'][key],3e-10)
for a,sigma in enumerate((1.,-1.)):
 k=1;h=.1*min(f[a,k],1-f[a,k])
 for sign in(-1.,1.):
  delta=sign*h;changed=f.copy();changed[a,k]+=delta
  newY=Y-sigma*g['weights'][k]*delta/n
  newu=matter['u']-g['weights'][k]*g['energies'][k]*delta/n
  newT=B.material.recover_temperature(newu,n,newY,T)
  totals=B.invariants(newT,newY,changed,n,g)
  gate(f'LTE local constrained entropy maximum {a}{sign}',max(totals['entropy_per_baryon']-initial['entropy_per_baryon'],0.),2e-12)
  gate(f'LTE perturbation energy {a}{sign}',totals['energy_MeV_per_baryon']-initial['energy_MeV_per_baryon'],3e-10)
  gate(f'LTE perturbation Le {a}{sign}',totals['electron_lepton_per_baryon']-initial['electron_lepton_per_baryon'],3e-13)
  samples.append(dict(flavor=a,delta_f=delta,newT=newT,newY=newY,entropy_change=totals['entropy_per_baryon']-initial['entropy_per_baryon']))
out=dict(passed=all(x['passed']for x in checks),checks=checks,samples=samples,runtime_seconds=time.monotonic()-start,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'thermal_box/box.py',ROOT/'thermal_box/results.json']})
(HERE/'lte_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if not out['passed']:raise SystemExit(1)
