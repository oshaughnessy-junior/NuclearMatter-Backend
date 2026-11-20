from pathlib import Path
import json,math,time,hashlib,atexit
import numpy as np
import redistribution as r
P=Path(__file__).resolve().parent;start=time.monotonic();rows=[];checks=[];phase='start';grid_gates=[]
paths=[Path(r.__file__),r.ROOT/'neutral_current/free_nc.py',r.ROOT/'response_reference/free_density_response.py',r.ROOT/'nr_baryon_eos/free_nucleons.py'];hashes={str(p.relative_to(r.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths}
def save():
 report=dict(passed=phase=='complete'and all(c['passed']for c in checks),phase=phase,checks=checks,grid_adequacy_gates=grid_gates,rows=rows,labels=r.LABELS,runtime_seconds=time.monotonic()-start,source_sha256=hashes,end_source_sha256={str(p.relative_to(r.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths})
 (P/'results.json').write_text(json.dumps(report,indent=2)+'\n')
atexit.register(save)
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and error<tol)))
def deadline():
 if time.monotonic()-start>105:raise TimeoutError('NCgrid105s experiment guard; partial results retained')
def errors(values,reference):
 v=np.array(values);ref=np.array(reference);scale=np.maximum(abs(ref),1e-30)
 # Signed energy/net separately scaled by their actual nonzero reference.
 return abs(v-ref)/scale
for label,T,n,E,rad in [('warm',2.,.002,10.,r.Radiation('FD',3.,4.)),('degenerate',.5,.005,20.,r.Radiation('packet',4.,5.)),('nearLTE',2.,.002,10.,r.Radiation('FD',2.00002,6.))]:
 deadline();phase=label;species=r.nc.state_density(T,n,0.);row=dict(label=label,state=species.asdict(),Ein=E,radiation=rad.__dict__,references=[],grids=[]);rows.append(row);save()
 for order,tail,coord in [(48,50.,'q'),(96,70.,'q'),(96,70.,'cos')]:
  deadline();row['references'].append(r.adaptive(species,E,rad,angle_order=order,tail=tail,coordinate=coord));save()
 ref=row['references'][1]['values']
 for i in [0,2]:
  errs=errors(row['references'][i]['values'],ref)
  for k,value in enumerate(errs):gate(f'{label} continuum reference{i} {r.LABELS[k]}',value,3e-5)
 for kind,orders in [('uniform',[32,64,128]),('resolved',[4,8,16,24])]:
  for order in orders:
   # Quadrature controls>=8 applies to kernel/angular; four-point outer Gauss allowed via order8 base? Intentionally use8 minimum below.
   if order==4:continue
   deadline();grid=r.energy_grid(species,E,rad,kind=kind,order=order,angle_order=96,tail=70.);errs=errors(grid['values'],ref);grid['relative_errors']=errs.tolist();row['grids'].append(grid)
   for k,error in enumerate(errs):grid_gates.append(dict(name=f'{label} {kind}{order} {r.LABELS[k]}',error=float(error),tolerance=3e-5,passed=bool(error<3e-5)))
   save()
 final=row['grids'][-1]
 for k,error in enumerate(final['relative_errors']):gate(f'{label} resolved24 accepted {r.LABELS[k]}',error,3e-5)
 # Physical reverse kernels and DOS ratio, not an imposed KMS inversion.
 pair=r.nc.pair_from_species(species);kms=[]
 for Ep in [E*.7,E*.999,E*1.001,E*1.3]:
  for c in [-.7,.2,.8]:
   forward=r.nc.log_kernel(pair,E,Ep,c);reverse=r.nc.log_kernel(pair,Ep,E,c)
   residual=reverse-forward-2*math.log(E/Ep)+(E-Ep)/T;kms.append(residual)
 gate(label+' independent bare reverse DOS KMS',max(abs(x)for x in kms),3e-11);row['reverse_DOS_residuals']=kms;save()
 print(label,'time',time.monotonic()-start,'failed',[c for c in checks if not c['passed']],flush=True)
phase='complete';save();print(len(checks),all(c['passed']for c in checks),time.monotonic()-start)
