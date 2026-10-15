from pathlib import Path
import sys,json,math,time,hashlib,atexit
import numpy as np
from numpy.polynomial.legendre import leggauss
import reference as F
P=Path(__file__).resolve().parent;R=P.parent;sys.dont_write_bytecode=True;sys.path.insert(0,str(R/'leptonic_pair_annihilation'));import pair as A
files=[x for x in (R/'leptonic_pair_annihilation').iterdir()if x.is_file()];pin={str(x.relative_to(R)):hashlib.sha256(x.read_bytes()).hexdigest()for x in files};rows=[];checks=[];start=time.monotonic();phase='vacuum'
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
def save():
 (P/'results.json').write_text(json.dumps(dict(phase=phase,rows=rows,checks=checks,passed=all(x['passed']for x in checks),seconds=time.monotonic()-start,source_sha256=pin,end_source_sha256={str(x.relative_to(R)):hashlib.sha256(x.read_bytes()).hexdigest()for x in files}),indent=2)+'\n')
atexit.register(save)
for flavor in ['e','mu']:
 c=A.Couplings(flavor)
 for ee in [.515,.743,3.17]:
  s=4*ee*ee;beta=math.sqrt(1-F.ME**2/ee**2)
  for angle in [-.71,.13,.82]:
   pm,pp,k,a=F.cm(s,angle);tr=F.trace(pm,pp,k,a,c.cv,c.ca);sc=F.scalar_cm(s,angle,c.cv,c.ca)
   gate('fresh trace vs ownCM polynomial',tr/sc-1,3e-11);gate('freshtrace vs authoramplitude',float(A.amplitude(*[z[None,:]for z in [pm,pp,k,a]],c)[0])/tr-1,3e-11)
  x,w=leggauss(24);sigma=sum(ww*F.scalar_cm(s,float(xx),c.cv,c.ca)/(32*math.pi*s*beta)for xx,ww in zip(x,w));sig=A.sigma(s,c);gate('fresh angular cross-section absolute',sig/sigma-1,3e-11)
 for T,mu in [(2.,.17446333200826397),(1.73,.29),(.37,.12)]:
  phase='empty'+str((flavor,T));ref=F.separated(T,mu,c.cv,c.ca);low=F.separated(T,mu,c.cv,c.ca,tail=50.);run=A.thermal(T,mu,c,radial=16,incident=8,outgoing=8,azimuth=8,tail=50.);rows.append(dict(kind='empty',flavor=flavor,T=T,mu=mu,reference=ref,author=run))
  for key in ['number','nu_energy','antinu_energy','electron_energy','positron_energy']:
   gate('1D reference tail '+str((T,key)),low[key]/ref[key]-1,2e-10);gate('fresh separated absolute '+str((flavor,T,key)),run['totals'][key]['value']/ref[key]-1,2e-6)
  save()
phase='complete';save();print(len(checks),all(c['passed']for c in checks),time.monotonic()-start);print([c for c in checks if not c['passed']])
