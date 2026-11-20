"""Independent q-outer, outgoing-energy-inner NC continuum review."""
from pathlib import Path
import sys,math,json,time,hashlib,atexit
import numpy as np
from scipy.special import log_expit
from numpy.polynomial.legendre import leggauss
P=Path(__file__).resolve().parent;ROOT=P.parent;sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'nc_continuum_grid'));import redistribution as r
start=time.monotonic();checks=[];rows=[];failures=[]
paths=list((ROOT/'nc_continuum_grid').glob('*.py'))+list((ROOT/'nc_continuum_grid').glob('*.json'))+[ROOT/'neutral_current/free_nc.py']
sha=lambda:{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
before=sha()
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error) and error<tol)))
def save():
 (P/'results.json').write_text(json.dumps(dict(checks=checks,rows=rows,failures=failures,source_hashes_before=before,source_hashes_after=sha(),runtime_seconds=time.monotonic()-start),indent=2)+'\n')
atexit.register(save)
def independent(sp,E,rad,cutoff,qorder,eorder):
 """Physical triangle domain q first; both forward/reverse response evaluated."""
 pair=r.nc.pair_from_species(sp);qx,qw=leggauss(qorder);ex,ew=leggauss(eorder);total=np.zeros(7)
 qedges=sorted(set([0.,.1*E,.3*E,E,2*E,3*E,E+cutoff]));qedges=[v for v in qedges if v<=E+cutoff]
 lfi=float(log_expit((rad.mu-E)/rad.T));lbi=float(log_expit((E-rad.mu)/rad.T))
 for qlo,qhi in zip(qedges[:-1],qedges[1:]):
  for x,w in zip(qx,qw):
   q=qlo+(x+1)*(qhi-qlo)/2;lo=abs(E-q);hi=min(cutoff,E+q)
   if lo>=hi:continue
   width=max(q*math.sqrt(sp.T/sp.mass),q*math.sqrt(2*max(sp.kinetic_mu,0)/sp.mass),q*q/(2*sp.mass));center=E-q*q/(2*sp.mass)
   es=sorted(set([lo,hi]+[center+k*width for k in (-12,-6,-3,-1,0,1,3,6,12) if lo<center+k*width<hi]+([E] if lo<E<hi else [])))
   partial=np.zeros(7)
   for elo,ehi in zip(es[:-1],es[1:]):
    for y,v in zip(ex,ew):
     Ep=elo+(y+1)*(ehi-elo)/2;c=(E*E+Ep*Ep-q*q)/(2*E*Ep);c=min(1.,max(-1.,c));omega=E-Ep
     lf=float(log_expit((rad.mu-Ep)/rad.T));lb=float(log_expit((Ep-rad.mu)/rad.T));lk=r.nc.log_kernel(pair,E,Ep,c);lr=r.nc.log_kernel(pair,Ep,E,c)
     blocked=math.exp(lk+lb);lgain=lr+2*math.log(Ep/E)+lf+lbi;lloss=lk+lfi+lb
     gain=math.exp(lgain);loss=math.exp(lloss)
     net=loss*math.expm1(lgain-lloss) if abs(lgain-lloss)<700 else gain-loss
     value=np.array([blocked,(1-c)*blocked,omega*blocked,abs(omega)*blocked,gain,loss,net])
     partial+=v*(ehi-elo)/2*q/(E*Ep)*value
   total+=w*(qhi-qlo)/2*partial
 return total
for label,T,n,E,rt,mu in [('heldout_warm',1.1,.0017,13.7,1.8,2.3),('heldout_cold',.2,.0042,19.6,.7,5.1),('heldout_nearLTE',1.1,.0017,13.7,1.1*(1+1e-7),3.2)]:
 sp=r.nc.state_density(T,n,0.);rad=r.Radiation('FD',rt,mu);cut=r.upper(sp,E,70.);row=dict(label=label,state=dict(T=T,n=n,E=E,radiationT=rt,mu=mu),cutoff=cut);rows.append(row)
 try:
  row['author_q96']=r.adaptive(sp,E,rad,angle_order=96,tail=70.);ref=np.array(row['author_q96']['values']);save()
  for qo,eo in ((32,20),(56,32)):
   val=independent(sp,E,rad,cut,qo,eo);row[f'independent_q{qo}_E{eo}']=val.tolist();save()
  val=np.array(row['independent_q56_E32']);coarse=np.array(row['independent_q32_E20'])
  for i,name in enumerate(r.LABELS):
   gate(label+' independent forward '+name,abs(val[i]/ref[i]-1),3e-5)
   gate(label+' independent refinement '+name,abs(coarse[i]/val[i]-1),3e-5)
  # Actually change absolute cutoff; nominal tail may be width-dominated.
  extended=independent(sp,E,rad,1.5*cut,56,32);row['absolute_cutoff_extended']=extended.tolist()
  for i,name in enumerate(r.LABELS):gate(label+' actual cutoff1.5 '+name,abs(extended[i]/val[i]-1),3e-5)
 except Exception as exc:row['exception']=repr(exc);failures.append(dict(label=label,error=repr(exc)))
 save()
 if time.monotonic()-start>105:break
# A colder valid-input probe: record broader-domain log/ordinary hazards honestly.
sp=r.nc.state_density(.08,.0042,0.);rad=r.Radiation('FD',.4,5.)
try:
 value=r.adaptive(sp,19.6,rad,angle_order=48,tail=70.);rows.append(dict(label='cold_domain_probe',result=value))
except Exception as exc:failures.append(dict(label='cold_domain_probe',T=.08,n=.0042,E=19.6,cutoff=r.upper(sp,19.6,70.),exception=repr(exc)))
gate('all author source/data hashes unchanged',float(sha()!=before),.1);save();print('checks',sum(c['passed'] for c in checks),'/',len(checks),'failures',failures,'runtime',time.monotonic()-start)
for c in checks:
 if not c['passed']:print('FAIL',c)
