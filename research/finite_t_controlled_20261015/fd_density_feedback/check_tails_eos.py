"""Independent large-energy panels and genuine EOS derivatives at actual states."""
from engine import *
start=time.monotonic();density_eos.set_budget(100.);checks=[];rows=[];provenance=physical_provenance()
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error) and error<tol)))
b=json.loads((P/'results_4.json').read_text());q=json.loads((P/'qualification.json').read_text())
if not b['passed'] or not q['passed'] or provenance!=b['provenance'] or provenance!=q['provenance']:raise ValueError('Require current accepted author forward data')
peak=max((r['before'] for r in b['rows']),key=lambda x:x['nB'])
for label,point in [('peak',peak),('endpoint',b['final'])]:
 T,n,Y=point['T'],point['nB'],point['Ye'];s=density_eos.state(T,n,Y)
 for divisor in (1e4,2e4):
  for key,val,analytic in [('T',T,s['u_T']),('n',n,s['u_n_MeV_fm3']),('Y',Y,s['u_Y'])]:
   h=val/divisor;args1=[T,n,Y];args2=[T,n,Y];index={'T':0,'n':1,'Y':2}[key];args1[index]+=h;args2[index]-=h
   numerical=(density_eos.state(*args1)['u_MeV_per_baryon']-density_eos.state(*args2)['u_MeV_per_baryon'])/(2*h)
   gate(label+' FD '+key+str(divisor),abs(numerical/analytic-1),5e-7)
 # Independent extension above the production 100MeV cutoff, no generic static grid.
 m=s['matter'];tailrows=[]
 for flavor in ('nue','antinue'):
  for order in (6,10):
   xs,ws=leggauss(order);total=np.zeros(2)
   for lo,hi in ((100.,160.),(160.,240.)):
    for x,w in zip(xs,ws):
     E=float((lo+hi)/2+(hi-lo)/2*x);c=element.physics.coefficients(m,flavor,E,order=48,tail=40.)
     N=element.DOS*E**2*c['j_s'];total+=w*(hi-lo)/2*np.array([N,E*N])
   tailrows.append(dict(flavor=flavor,order=order,number=total[0],energy=total[1]))
   # Scale tails separately by same-flavor whole moments (peak actual source or endpoint fine).
   ref=q['details']['endpoint_source10']['result'] if label=='endpoint' else max((r['result'] for r in b['sources']),key=lambda a:a['nB'])
   for index,field in enumerate(('number_fm3_s','energy_MeV_fm3_s')):gate(label+' neglected100-240 '+flavor+field+str(order),abs(total[index]/ref['moments'][flavor][field]),1e-12)
 rows.append(dict(label=label,state=point,tails=tailrows));(P/'tails_partial.json').write_text(json.dumps(dict(checks=checks,rows=rows),indent=2)+'\n')
gate('physical provenance unchanged',float(provenance!=physical_provenance()),.1)
report=dict(passed=all(c['passed'] for c in checks),checks=checks,rows=rows,provenance=provenance,runtime_seconds=time.monotonic()-start,scope='Selected actual states; no bound on mathematical E>240 tail')
(P/'tails_eos.json').write_text(json.dumps(report,indent=2)+'\n');print(sum(c['passed'] for c in checks),'/',len(checks),report['passed'],report['runtime_seconds'])
for c in checks:
 if not c['passed']:print('FAIL',c)
