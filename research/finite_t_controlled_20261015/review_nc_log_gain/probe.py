"""Fresh q-outer/Eout-inner direct-reverse NC gain review, no author edits."""
from pathlib import Path
import sys,math,json,time,hashlib
import numpy as np
from scipy.special import log_expit
from numpy.polynomial.legendre import leggauss
ROOT=Path(__file__).resolve().parent.parent;HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'nc_log_gain'))
import log_gain as R
start=time.monotonic();checks=[];details={}
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and abs(error)<=tol)))

def separate(sp,E,rad,cut,qo,eo):
 # Triangle domain inverted relative to author: q outer; Ep inner.
 xs,ws=leggauss(qo);ys,vs=leggauss(eo);pair=R.nc.pair_from_species(sp);tot=np.zeros(7)
 lfi=float(log_expit((rad.mu-E)/rad.T));lvi=float(log_expit((E-rad.mu)/rad.T))
 qedges=sorted({0.,.1*E,.3*E,E,2*E,3*E,E+cut})
 for a,b in zip(qedges[:-1],qedges[1:]):
  for x,wx in zip(xs,ws):
   if time.monotonic()-start>105:raise TimeoutError('Review105s numerical bound')
   q=a+(x+1)*(b-a)/2;lo,hi=abs(E-q),min(cut,E+q)
   if lo>=hi:continue
   width=max(q*math.sqrt(sp.T/sp.mass),q*math.sqrt(2*max(sp.kinetic_mu,0.)/sp.mass),q*q/(2*sp.mass))
   center=E-q*q/(2*sp.mass)
   es=sorted({lo,hi}|{v for k in(-12,-6,-3,-1,0,1,3,6,12)if lo<(v:=center+k*width)<hi}|({E}if lo<E<hi else set()))
   for c,d in zip(es[:-1],es[1:]):
    for y,wy in zip(ys,vs):
     Ep=c+(y+1)*(d-c)/2;mu=(E*E+Ep*Ep-q*q)/(2*E*Ep);mu=min(1.,max(-1.,mu));omega=E-Ep
     lf=float(log_expit((rad.mu-Ep)/rad.T));lv=float(log_expit((Ep-rad.mu)/rad.T))
     lk=R.nc.log_kernel(pair,E,Ep,mu);lr=R.nc.log_kernel(pair,Ep,E,mu)
     gainlog=lr+2*math.log(Ep/E)+lf+lvi;losslog=lk+lfi+lv
     gain,loss=math.exp(gainlog),math.exp(losslog);blocked=math.exp(lk+lv)
     # Independent reverse rate. expm1 used only after complete gain/loss logs.
     delta=gainlog-losslog
     net=loss*math.expm1(delta)if abs(delta)<600 else gain-loss
     value=np.array([blocked,(1-mu)*blocked,omega*blocked,abs(omega)*blocked,gain,loss,net])
     tot+=wx*(b-a)/2*wy*(d-c)/2*q/(E*Ep)*value
 return tot

sp=R.nc.state_density(.08,.0042,0.);E=19.6;rad=R.Radiation('FD',.4,5.);cut=R.original.upper(sp,E,70.)
fine=json.loads((ROOT/'nc_log_gain/cold_refinement.json').read_text())['runs'][2]['values']
runs=[]
for qo,eo in[(160,80),(192,96)]:
 values=separate(sp,E,rad,cut,qo,eo);runs.append(values)
 for k,name in enumerate(R.original.LABELS):gate(f'new coordinate{qo} '+name,values[k]/fine[k]-1,3e-5)
details['independent_cold']=[dict(qorder=qo,eorder=eo,values=v.tolist())for(qo,eo),v in zip([(160,80),(192,96)],runs)]
for k,name in enumerate(R.original.LABELS):gate('new independent refinement '+name,runs[0][k]/runs[1][k]-1,3e-5)
extended=separate(sp,E,rad,1.5*cut,160,80)
for k,name in enumerate(R.original.LABELS):gate('new coordinate cutoff1.5 '+name,extended[k]/runs[0][k]-1,1e-8)
details['independent_cold_extended']=dict(cutoff=1.5*cut,qorder=160,eorder=80,values=extended.tolist())
main=json.loads((ROOT/'nc_log_gain/results.json').read_text());warm=[]
for label,rt,mu in[('warm',3.,4.),('nearLTE',2.00002,6.)]:
 species=R.nc.state_density(2.,.002,0.);radiation=R.Radiation('FD',rt,mu)
 values=separate(species,10.,radiation,R.original.upper(species,10.,70.),64,40)
 ref=next(row for row in main['rows']if row['label']==label)['runs'][2]['values']
 for k,name in enumerate(R.original.LABELS):gate('new coordinate '+label+' '+name,values[k]/ref[k]-1,3e-5)
 warm.append(dict(label=label,values=values.tolist(),reference=ref))
details['independent_warm_regressions']=warm

# Actual original exception and finite physical logs at the exact witness.
Ep=77.2
try:R.original.integrand(sp,E,Ep,rad)
except OverflowError:rejected=True
else:rejected=False
gate('original overflow still present',0. if rejected else 1.,0.)
pair=R.nc.pair_from_species(sp);lf=R.nc.log_kernel(pair,E,Ep,-.3);lr=R.nc.log_kernel(pair,Ep,E,-.3)
loggain=lr+2*math.log(Ep/E)+float(log_expit((rad.mu-Ep)/rad.T)+log_expit((E-rad.mu)/rad.T))
point=R.point(sp,E,Ep,-.3,rad,reverse='direct')
gate('direct response DOS KMS',lr-lf-2*math.log(E/Ep)+(E-Ep)/sp.T,3e-10)
gate('independent combined gain log',point['log_gain']-loggain,3e-10)
gate('underflow finite log not support zero',0. if math.isfinite(loggain)and point['gain']['value']is None and point['gain']['reason']=='underflow'else 1.,0.)
details['witness']=dict(log_forward=lf,log_reverse=lr,log_combined_gain=loggain,point=point)

# Independent point occupation factors, including near LTE and non-FD packet.
for r in[R.Radiation('FD',sp.T,6.),R.Radiation('FD',sp.T*(1+1e-7),6.),R.Radiation('packet',.4,5.)]:
 for Ep in(18.9,20.3):
  p=R.point(sp,E,Ep,.27,r,reverse='direct');expected=R.point(sp,E,Ep,.27,r,reverse='kms')
  gate(f'actualreverse point {r.kind}T{r.T}Ep{Ep}',p['log_gain']-expected['log_gain'],3e-10)
  if r.T==sp.T and r.kind=='FD':gate('arbitrary mu LTE point',p['net']['value']/(p['gain']['value']+p['loss']['value']),3e-12)

# Check the archived coarse failures retain their failed status and magnitude.
main=json.loads((ROOT/'nc_log_gain/results.json').read_text());failed=[c for c in main['checks']if not c['passed']]
gate('two original coarse gates remain failed',0. if len(failed)==2 and not main['passed']and all(c['error']>c['tolerance']==3e-5 for c in failed)else 1.,0.)
details['preserved_failed_gates']=failed
out=dict(passed=all(c['passed']for c in checks),checks=checks,details=details,runtime_seconds=time.monotonic()-start,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in[Path(__file__),ROOT/'nc_log_gain/log_gain.py',ROOT/'nc_log_gain/results.json',ROOT/'nc_log_gain/cold_refinement.json',ROOT/'nc_continuum_grid/redistribution.py',ROOT/'neutral_current/free_nc.py',ROOT/'response_reference/free_density_response.py',ROOT/'review_nc_continuum/overflow_log_evidence.json']})
(HERE/'results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(passed=out['passed'],checks=len(checks),seconds=out['runtime_seconds'],failed=[x for x in checks if not x['passed']])))
if not out['passed']:raise SystemExit(1)
