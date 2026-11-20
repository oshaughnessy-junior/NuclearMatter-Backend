"""Bounded new-cutoff/global-source qualification; no evolving material."""
from exchange import *
import json,atexit
P=Path(__file__).resolve().parent;start=time.monotonic();checks=[];rows=[]
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error) and error<tol)))
def save():(P/'qualification.json').write_text(json.dumps(dict(checks=checks,rows=rows,runtime_seconds=time.monotonic()-start),indent=2)+'\n')
atexit.register(save)
base=json.loads((P/'initial_bounded_partial.json').read_text());lte=json.loads((P/'results_LTE.json').read_text());cases=[r for r in base['rows'] if r['label']!='LTE']+lte['rows'];sp=nc.state_density(2.,.002,0.)
for row in cases:
 label=row['label'];ref=np.array(row['runs']['paired_ref']['values']);gross=ref[5]+ref[6]
 for name,run in row['runs'].items():
  gate(label+name+' exact current sourcehashes',float(run['source_hashes']!=hashes()),.1)
  gate(label+name+' exact FD bathstate',float(run['bath']!=sp.asdict()),.1)
  gate(label+name+' exact prescribed radiation',float(run['radiation']!=row['radiation']),.1)
 for name in ('coarse','energy_only','angle_only'):
  v=np.array(row['runs'][name]['values']);gate(label+name+' positive gross eventcount',abs((v[5]+v[6])/gross-1),3e-5)
 direct=np.array(row['runs']['direct']['values']);gate(label+' independent positive forwardeventcount',abs(direct[5]/gross-1),3e-5);gate(label+' independent positive reverseeventcount',abs(direct[6]/gross-1),3e-5)
 if not row['radiation']['packet']:
  rt=row['radiation']['T'];gate(label+' FD radiation entropy flux',abs(ref[2]+ref[3]/rt)/gross,3e-12);gate(label+' FD bath+radiation Hidentity',abs(ref[4]-ref[3]*(1/sp.T-1/rt))/gross,3e-12)
 if label in ('hot','packet'):
  rad=Radiation(**row['radiation']);record=dict(label=label);rows.append(record)
  record['paired144']=integrate(sp,rad,cutoff=144.,energy_order=12,inner_order=24,angle_order=48);save()
  record['direct108']=integrate(sp,rad,method='direct',cutoff=108.,energy_order=16,inner_order=32,angle_order=96);save()
  ref108=np.array(row['runs']['cutoff108']['values']);v144=np.array(record['paired144']['values']);dd=np.array(record['direct108']['values'])
  for idx in (1,2,3,4):
   scale=max(abs(ref108[idx]),gross*1e-8)
   gate(label+' cutoff108->144 '+str(idx),abs(v144[idx]-ref108[idx])/scale,3e-5)
   gate(label+' independent direct108 '+str(idx),abs(dd[idx]-ref108[idx])/scale,3e-5)
  gate(label+' direct108 number source',abs(dd[0])/gross,3e-6)
  # Upper-upper box explicitly closes the finite-box correction decomposition.
  x,w=nodes(12);y,v=nodes(24);total=np.zeros(4)
  for xx,ww in zip(x,w):
   E=72.+(xx+1)*18.;lf,lb=rad.logs(E);hp=lb-lf
   for a,b in zip(partner_edges(sp,E,E,108.)[:-1],partner_edges(sp,E,E,108.)[1:]):
    for yy,vv in zip(y,v):
     Ep=a+(yy+1)*(b-a)/2;J,F,R,H=events(sp,E,Ep,rad,48,'q');lp,bp=rad.logs(Ep)
     total+=ww*18.*vv*(b-a)/2*np.array([(Ep-E)*J,((bp-lp)-hp)*J,(E-Ep)*J,H])
  record['upper_upper_72_108']=total.tolist();cross=np.array(row['runs']['crossing72_108']['values']);before=np.array(row['runs']['paired_ref']['values'])
  for j,idx in enumerate((1,2,3,4)):gate(label+' independent finitebox decomposition '+str(idx),abs(ref108[idx]-before[idx]-cross[idx]-total[j])/max(abs(ref108[idx]),gross*1e-8),3e-6)
 save()
 if time.monotonic()-start>105:raise TimeoutError('Qualification105s guard')
print('checks',sum(c['passed']for c in checks),'/',len(checks),'runtime',time.monotonic()-start)
for c in checks:
 if not c['passed']:print('FAIL',c)
