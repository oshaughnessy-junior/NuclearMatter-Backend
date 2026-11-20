"""Independent log-incident-energy full-square/block NC exchange review."""
from pathlib import Path
import sys,json,math,time,hashlib,atexit
import numpy as np
from scipy.special import expit,log_expit
from numpy.polynomial.legendre import leggauss
from functools import lru_cache
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent;sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'nc_global_exchange'))
import exchange as author
nc=author.nc;start=time.monotonic();checks=[];raw={};phase='start';paths=[ROOT/'nc_global_exchange'/x for x in ['exchange.py','qualification.json','results_LTE.json','initial_budget_stop.txt']]+[ROOT/'neutral_current/free_nc.py',ROOT/'response_reference/free_density_response.py'];hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths}
def save():
 end={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths}
 (HERE/'results.json').write_text(json.dumps(dict(passed=phase=='complete'and all(c['passed']for c in checks)and hashes==end,phase=phase,checks=checks,raw=raw,runtime_seconds=time.monotonic()-start,source_sha256=hashes,end_source_sha256=end),indent=2)+'\n')
atexit.register(save)
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and error<tol)))
sp=nc.state_density(1.7,.0019,0.);trad,munu=2.35,2.8;rad=author.Radiation(trad,munu);pair=nc.pair_from_species(sp);nodes=lru_cache(maxsize=12)(leggauss)
def integrate(lo,hi,plo,phi,eo,po,ao,cross=False):
 # Incoming energy uses log(1+E); outgoing energy uses direct E, unrelated orders.
 xs,ws=nodes(eo);ys,vs=nodes(po);cs,cws=nodes(ao);total=np.zeros(7);panels=[]
 ees=sorted({lo,hi}|{x for x in [3.,7.,13.,22.,31.]if lo<x<hi})
 for a,b in zip(ees[:-1],ees[1:]):
  partial=np.zeros(7);za,zb=math.log1p(a),math.log1p(b)
  for x,w in zip(xs,ws):
   z=za+(x+1)*(zb-za)/2;E=math.expm1(z);weightE=w*(zb-za)/2*math.exp(z);lfi=float(log_expit((munu-E)/trad));lbi=float(log_expit((E-munu)/trad));hp=lbi-lfi
   width=E*max(math.sqrt(sp.T/sp.mass),math.sqrt(2*max(sp.kinetic_mu,0.)/sp.mass),E/(2*sp.mass))
   edges=sorted({plo,phi}|{E+k*width for k in [-16,-8,-4,-2,-1,-.4,-.1,0.,.1,.4,1,2,4,8,16]if plo<E+k*width<phi})
   for c,d in zip(edges[:-1],edges[1:]):
    for y,v in zip(ys,vs):
     Ep=c+(y+1)*(d-c)/2;weight=weightE*v*(d-c)/2;lfo=float(log_expit((munu-Ep)/trad));lbo=float(log_expit((Ep-munu)/trad));Hpo=lbo-lfo
     # Actual reverse kernel with its own physical E² and explicit DOS(E').
     J=A=B=0.
     for cos,cw in zip(cs,cws):
      la=math.log(nc.DOS)+2*math.log(E)+nc.log_kernel(pair,E,Ep,float(cos))+lfi+lbo
      lb=math.log(nc.DOS)+2*math.log(Ep)+nc.log_kernel(pair,Ep,E,float(cos))+lfo+lbi
      aa,bb=math.exp(la),math.exp(lb);jj=aa*(-math.expm1(lb-la))if la>=lb else -bb*(-math.expm1(la-lb))
      J+=cw*jj;A+=cw*aa;B+=cw*bb
     if cross:
      q=(E-Ep)*J;srad=(Hpo-hp)*J;val=np.array([-J,-q,srad,q,srad+q/sp.T,A,B])
     else:val=np.array([-J,-E*J,-hp*J,E*J,(-hp+E/sp.T)*J,A,B])
     partial+=weight*val
   if time.monotonic()-start>110:raise TimeoutError('Independent review110s budget; completed panels saved')
  total+=partial;panels.append(dict(lo=a,hi=b,values=partial.tolist()));raw['latest_partial']=dict(bounds=[lo,hi,plo,phi],orders=[eo,po,ao],panels=panels,total=total.tolist());save()
 return dict(values=total.tolist(),bounds=[lo,hi,plo,phi],orders=[eo,po,ao],cross_pair=cross,panels=panels)
phase='newheldout_lowbox'
raw['independent_low_coarse']=integrate(0.,24.,0.,24.,8,16,32);save()
raw['independent_low_fine']=integrate(0.,24.,0.,24.,12,24,64);save()
raw['paired_low_reference']=author.integrate(sp,rad,method='paired',cutoff=24.,energy_order=16,inner_order=32,angle_order=96,budget=45.);save()
coarse=np.array(raw['independent_low_coarse']['values']);fine=np.array(raw['independent_low_fine']['values']);ref=np.array(raw['paired_low_reference']['values']);gross=ref[5]+ref[6]
for k,label in [(1,'radiation energy'),(2,'radiation entropy'),(3,'bath heat'),(4,'total entropy')]:
 gate('newheldout independent refinement '+label,abs(coarse[k]/fine[k]-1),3e-5);gate('newheldout independent vs paired '+label,abs(fine[k]/ref[k]-1),3e-5)
gate('newheldout nonenforced number residual',abs(fine[0])/gross,3e-7)
gate('newheldout FD entropy heat identity',abs(fine[2]+fine[3]/trad)/max(abs(fine[2]),abs(fine[3]/trad)),3e-7)
gate('newheldout energy bath/rad opposite',abs(fine[1]+fine[3])/abs(fine[3]),1e-14)
gate('newheldout entropy nonnegative',float(fine[4]<0),.5)
phase='boundary_blocks';raw['independent_cross']=integrate(0.,24.,24.,36.,12,24,64,True);save();raw['independent_upper']=integrate(24.,36.,24.,36.,12,24,64);save();raw['paired_full36']=author.integrate(sp,rad,method='paired',cutoff=36.,energy_order=16,inner_order=32,angle_order=96,budget=45.);save()
full=np.array(raw['paired_full36']['values']);cross=np.array(raw['independent_cross']['values']);upper=np.array(raw['independent_upper']['values']);assembled=fine+cross+upper
for k,label in [(1,'radiation energy'),(2,'radiation entropy'),(3,'bath heat'),(4,'total entropy')]:gate('independent LL+LH+HH '+label,abs(assembled[k]/full[k]-1),3e-5)
raw['boundary_summary']=dict(retained_low_number_crossing=float(cross[0]),closed_low_number_residual=float(fine[0]),heat_low=float(fine[3]),heat_full=float(full[3]),assembled_heat=float(assembled[3]),heat_cutoff_change=float(full[3]-fine[3]));phase='complete';save();print(len(checks),all(c['passed']for c in checks),time.monotonic()-start);print([c for c in checks if not c['passed']])
