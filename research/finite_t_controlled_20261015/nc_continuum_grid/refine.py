from pathlib import Path
import json,math,time,atexit
import numpy as np
from scipy.integrate import quad_vec
import redistribution as r
P=Path(__file__).resolve().parent;initial=json.loads((P/'results.json').read_text());start=time.monotonic();checks=[];rows=[];phase='refinements'
def save():
 (P/'refinements.json').write_text(json.dumps(dict(passed=phase=='complete'and all(c['passed']for c in checks),phase=phase,checks=checks,rows=rows,runtime_seconds=time.monotonic()-start),indent=2)+'\n')
atexit.register(save)
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and error<tol)))
for old in initial['rows']:
 sp=r.nc.state_density(old['state']['T'],old['state']['density'],old['state']['offset']);E=old['Ein'];rad=r.Radiation(**old['radiation']);ref=np.array(old['references'][1]['values']);label=old['label'];row=dict(label=label);rows.append(row)
 for name,kw in [('tail_only',dict(angle_order=96,tail=50.)),('angular_only',dict(angle_order=64,tail=70.))]:
  val=r.adaptive(sp,E,rad,**kw);row[name]=val
  for k,error in enumerate(abs(np.array(val['values'])/ref-1)):gate(label+name+r.LABELS[k],error,3e-5)
 fine=r.energy_grid(sp,E,rad,order=32,angle_order=96,tail=70.);row['resolved32']=fine
 for k,error in enumerate(abs(np.array(fine['values'])/ref-1)):gate(label+' resolved32 '+r.LABELS[k],error,3e-5)
 save()
 # Reverse gain is computed DIRECTLY from reverse response, with outgoing DOS ratio.
 pair=r.nc.pair_from_species(sp);cs,ws=r.nodes_cached(96);fi=rad.f(E)
 def direct(Ep,rr=rad):
  if Ep<=0:return np.zeros(3)
  fo=rr.f(Ep);fin=rr.f(E);gain=loss=0.
  for c,w in zip(cs,ws):
   gain+=w*(Ep/E)**2*r.nc.kernel(pair,Ep,E,float(c))*fo*(1-fin)
   loss+=w*r.nc.kernel(pair,E,Ep,float(c))*fin*(1-fo)
  return np.array([gain,loss,gain-loss])
 total=np.zeros(3);error=0.;edges=r.edges(sp,E,r.upper(sp,E,70.))
 for lo,hi in zip(edges[:-1],edges[1:]):
  value,err=quad_vec(direct,lo,hi,epsabs=1e-12,epsrel=2e-10,limit=250);total+=value;error+=err
 row['direct_reverse_gain_loss_net']=total.tolist();row['direct_quad_absolute_error']=error
 for k,idx in enumerate([4,5,6]):gate(label+' actualreverse integrand '+r.LABELS[idx],abs(total[k]/ref[idx]-1),3e-5)
 if label=='nearLTE':
  thermal=r.Radiation('FD',sp.T,rad.mu);thermal_net=0.;gross=0.
  for lo,hi in zip(edges[:-1],edges[1:]):
   value,err=quad_vec(lambda x:direct(x,thermal),lo,hi,epsabs=1e-12,epsrel=2e-10,limit=250);thermal_net+=value[2];gross+=value[0]+value[1]
  gate('exactLTE actualreverse grossscaled residual',abs(thermal_net)/gross,3e-12);row['exactLTE_direct_net']=thermal_net;row['exactLTE_gross']=gross
 save()
 if time.monotonic()-start>105:raise TimeoutError('Refinement105s bound reached')
phase='complete';save();print(len(checks),all(c['passed']for c in checks),time.monotonic()-start);print([c for c in checks if not c['passed']])
