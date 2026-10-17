from pathlib import Path
import sys,json,math,time
sys.dont_write_bytecode=True
from events import *
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'muon_full_current'));import current as C
start=time.monotonic();b=DILUTE;ch=C.V.Channel(b.ma,b.mb,b.ml,b.anti);x,wx=leggauss(14);c,wc=leggauss(14);total=0.;density=0.
# Separate collision-flux integral using target-rest invariant total vacuum cross section.
# It never constructs final bath-frame states or uses this leaf's amplitude routine.
for lo,hi in zip([0,1,3,8,16,30],[1,3,8,16,30,60]):
 slo,shi=math.sqrt(lo*b.T),math.sqrt(hi*b.T)
 for z,w in zip((slo+shi)/2+(shi-slo)*x/2,wx*(shi-slo)/2):
  k=z*z;E=b.ma+k;p=math.sqrt(k*(k+2*b.ma));f=math.exp(float(log_expit((b.mua-E)/b.T)));measure=p*E*2*z*w*f/(math.pi**2*HC**3);density+=measure
  for ci,wi in zip(c,wc):
   flux=1-p*ci/E;rest=b.Enu*(E-p*ci)/b.ma;sigma_cm2=C.total(ch,rest,order=16,method='source');total+=measure*wi/2*flux*sigma_cm2*1e26/HC**2*HC**3/HBAR
r=json.loads(Path(__file__).with_name('results.json').read_text());event=[z for z in r['rows']if z['label']=='dilute'][-1]['result'];checks=[]
err=total/event['chi']['value']-1;checks.append(dict(name='dilute independent collision-flux cσ anchor',error=err,tolerance=3e-7,passed=abs(err)<3e-7));err=density/7.000000000001131e-16-1;checks.append(dict(name='anchor independent target density',error=err,tolerance=3e-9,passed=abs(err)<3e-9))
stationary=7.000000000001131e-16*1e39*2.99792458e10*C.total(ch,150,order=32,method='source')
Path(__file__).with_name('anchor_results.json').write_text(json.dumps(dict(passed=all(c['passed']for c in checks),checks=checks,collision_flux_chi=total,event_chi=event['chi']['value'],stationary_target_chi=stationary,thermal_correction=total/stationary-1,runtime_seconds=time.monotonic()-start),indent=2,default=lambda x:x.item())+'\n');print(checks,time.monotonic()-start)
