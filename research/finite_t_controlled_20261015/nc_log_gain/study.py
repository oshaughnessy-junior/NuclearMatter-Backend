from pathlib import Path
import json,math,time,hashlib,atexit
import numpy as np
import log_gain as r
P=Path(__file__).resolve().parent;start=time.monotonic();checks=[];rows=[];phase='start';paths=[Path(r.__file__),r.ROOT/'nc_continuum_grid/redistribution.py',r.ROOT/'neutral_current/free_nc.py',r.ROOT/'response_reference/free_density_response.py',r.ROOT/'review_nc_continuum/overflow_log_evidence.json',r.ROOT/'review_nc_continuum/refined_results.json'];hashes={str(p.relative_to(r.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths}
def save():
 (P/'results.json').write_text(json.dumps(dict(passed=phase=='complete'and all(c['passed']for c in checks),phase=phase,checks=checks,rows=rows,runtime_seconds=time.monotonic()-start,source_sha256=hashes,end_source_sha256={str(p.relative_to(r.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths}),indent=2)+'\n')
atexit.register(save)
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and error<tol)))
def deadline():
 if time.monotonic()-start>105:raise TimeoutError('LogNC105s experiment guard; partial saved')
sp=r.nc.state_density(.08,.0042,0.);E=19.6;rad=r.Radiation('FD',.4,5.);witness=E+720*sp.T
try:r.original.integrand(sp,E,witness,rad)
except OverflowError as exc:oldfailure=dict(type=type(exc).__name__,message=str(exc));gate('original overflow reproduced unchanged',0.,.5)
else:oldfailure=None;gate('original overflow reproduced unchanged',1.,.5)
point={mode:r.point(sp,E,witness,-.3,rad,reverse=mode)for mode in ['kms','direct']};rows.append(dict(label='exact_overflow_witness',T=sp.T,n=sp.density,Ein=E,Eout=witness,cosine=-.3,old_failure=oldfailure,points=point))
gate('witness finite combined loggain',float(not math.isfinite(point['kms']['log_gain'])),.5);gate('witness kms/direct loggain',abs(point['kms']['log_gain']-point['direct']['log_gain']),3e-10)
gate('witness ordinary underflow explicitly flagged',float(point['kms']['gain']['representable']or point['kms']['gain']['value']is not None),.5);save()
for label,T,n,E,rad in [('warm',2.,.002,10.,r.Radiation('FD',3.,4.)),('cold_repair',.08,.0042,19.6,r.Radiation('FD',.4,5.)),('nearLTE',2.,.002,10.,r.Radiation('FD',2.00002,6.))]:
 phase=label;sp=r.nc.state_density(T,n,0.);cut=r.original.upper(sp,E,70.);row=dict(label=label,T=T,n=n,Ein=E,radiation=rad.__dict__,runs=[]);rows.append(row);save()
 for eo,ao,mode,factor in [(48,144,'kms',1.),(64,192,'kms',1.),(96,288,'kms',1.),(96,288,'direct',1.),(96,288,'kms',1.5)]:
  deadline();value=r.integrate(sp,E,rad,energy_order=eo,angle_order=ao,cutoff=cut*factor,reverse=mode);row['runs'].append(value);save()
 ref=np.array(row['runs'][2]['values'])
 for index in [0,1,3,4]:
  val=np.array(row['runs'][index]['values'])
  for k,error in enumerate(abs(val/ref-1)):gate(label+f' comparison{index} '+r.original.LABELS[k],error,3e-5 if index!=4 else 1e-8)
 if label in ['warm','nearLTE']:
  old=json.loads((r.ROOT/'nc_continuum_grid/results.json').read_text());oldref=np.array(next(x for x in old['rows']if x['label']==label)['references'][1]['values'])
 else:
  old=json.loads((r.ROOT/'review_nc_continuum/refined_results.json').read_text());oldref=np.array(next(x for x in old['rows']if x['label']=='cold08 independent144')['values'])
 for k,error in enumerate(abs(ref/oldref-1)):gate(label+' archived different-coordinate reference '+r.original.LABELS[k],error,3e-5)
 row['archived_reference']=oldref.tolist();save();print(label,'runtime',time.monotonic()-start,'failed',[c for c in checks if not c['passed']],flush=True)
phase='complete';save();print(len(checks),all(c['passed']for c in checks),time.monotonic()-start)
