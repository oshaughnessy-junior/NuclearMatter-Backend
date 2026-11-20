"""Separate bounded endpoint-source, work and step-sensitivity qualification."""
from engine import *
import atexit
start=time.monotonic();density_eos.set_budget(112.);checks=[];detail={};failure=None
provenance=physical_provenance()
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error) and error<tol)))
def save():
 (P/'qualification_partial.json').write_text(json.dumps(dict(checks=checks,details=detail,failure=failure,runtime_seconds=time.monotonic()-start),indent=2)+'\n')
atexit.register(save)
a=json.loads((P/'results_2.json').read_text());b=json.loads((P/'results_4.json').read_text())
for report in (a,b):
 gate(f'{report["steps"]} step original gate',float(not report['passed']),.1)
 gate(f'{report["steps"]} current exact provenance',float(report['provenance']!=provenance),.1)
 for source in report['sources']:
  gate('request fingerprint integrity',float(source['fingerprint']!=digest(source['request'])),.1)
  gate('request actual state and controls',float(any(source['result'][k]!=source['request'][k] for k in ('T','nB','Ye','outer_order','internal_order','tail','edges'))),.1)
if not all(c['passed'] for c in checks):raise RuntimeError('Forward/provenance gate failed')
detail['step_refinement']={k:dict(two=a['final'][k],four=b['final'][k],absolute_difference=abs(a['final'][k]-b['final'][k])) for k in ('T','Ye','u')}
# These are diagnostics of this deliberately short path, not an order certificate.
for k,tol in [('T',1e-4),('Ye',2e-4),('u',5e-3)]:gate('2/4 step sensitivity '+k,detail['step_refinement'][k]['absolute_difference'],tol)
try:
 point=b['final'];T,n,Y=point['T'],point['nB'],point['Ye'];save()
 coarse=source_record(T,n,Y,provenance,outer=6);detail['endpoint_source6']=coarse;save()
 fine=source_record(T,n,Y,provenance,outer=10);detail['endpoint_source10']=fine;save()
 for fl in ('nue','antinue'):
  for field in ('number_fm3_s','energy_MeV_fm3_s'):
   gate('endpoint outer6->10 '+fl+field,abs(coarse['result']['moments'][fl][field]/fine['result']['moments'][fl][field]-1),3e-6)
 # New peak-density physical spectral nodes: internal controls independent of outer energy.
 peak=max((r['before'] for r in b['rows']),key=lambda x:x['nB']);m=density_eos.state(peak['T'],peak['nB'],peak['Ye'])['matter']
 detail['internal_nodes']=[]
 for fl in ('nue','antinue'):
  for energy in (4.,16.):
   x=element.physics.coefficients(m,fl,energy,order=48,tail=40.);y=element.physics.coefficients(m,fl,energy,order=72,tail=50.)
   detail['internal_nodes'].append(dict(state=peak,flavor=fl,energy=energy,coarse=x,fine=y))
   for key in ('log_j','log_chi'):gate('peak internal48->72 '+fl+str(energy)+key,abs(math.expm1(x[key]-y[key])),3e-6)
 save();detail['work_refinement']=[]
 for report in (a,b):
  for i,row in enumerate(report['rows']):
   kick=row['weak_kick'];W8=pressure_work(kick['T'],kick['nB'],kick['Ye'],row['after']['nB'],order=8)
   detail['work_refinement'].append(dict(steps=report['steps'],index=i,order4=row['work_pressure'],order8=W8))
   gate(f'{report["steps"]}step{i} work4->8',abs(W8-row['work_pressure']),3e-10)
   save()
 # Contract counterexamples reject before any physical source call.
 wrong=json.loads(json.dumps(provenance));wrong['source_sha256']['fd_density_feedback/engine.py']='0'*64
 for label,call in [('stale source',lambda:source_record(T,n,Y,wrong)),('changed-state initial',lambda:source_record(T,n,Y,provenance,initial=True)),('changed-controls initial',lambda:source_record(2.,1e-4,.3,provenance,initial=True,outer=10))]:
  try:call();rejected=False
  except ValueError:rejected=True
  gate('reject '+label,float(not rejected),.1)
except (RuntimeError,ValueError,ArithmeticError,TimeoutError) as exc:failure=dict(type=type(exc).__name__,message=str(exc))
gate('provenance stable during qualification',float(provenance!=physical_provenance()),.1)
report=dict(passed=failure is None and all(c['passed'] for c in checks),checks=checks,details=detail,failure=failure,provenance=provenance,runtime_seconds=time.monotonic()-start)
(P/'qualification.json').write_text(json.dumps(report,indent=2)+'\n');print(sum(c['passed'] for c in checks),'/',len(checks),'passed',report['passed'],'runtime',report['runtime_seconds'],'failure',failure)
for c in checks:
 if not c['passed']:print('FAIL',c)
