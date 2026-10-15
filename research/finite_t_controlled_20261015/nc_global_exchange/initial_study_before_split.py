from exchange import *
import json,atexit
P=Path(__file__).resolve().parent;start=time.monotonic();checks=[];rows=[];failed=[]
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error) and error<tol)))
def save():
 (P/'results.json').write_text(json.dumps(dict(checks=checks,rows=rows,runtime_seconds=time.monotonic()-start),indent=2)+'\n')
atexit.register(save)
sp=nc.state_density(2.,.002,0.)
for label,rad in [('hot',Radiation(3.,4.)),('cold',Radiation(1.5,2.)),('packet',Radiation(3.,4.,True)),('LTE',Radiation(2.,4.))]:
 row=dict(label=label,radiation=rad.metadata(),runs={});rows.append(row)
 for name,kwargs in [('coarse',dict(energy_order=4,inner_order=6,angle_order=8)),('paired_ref',dict(energy_order=12,inner_order=24,angle_order=48)),('energy_only',dict(energy_order=20,inner_order=40,angle_order=48)),('angle_only',dict(energy_order=12,inner_order=24,angle_order=96)),('direct',dict(method='direct',energy_order=16,inner_order=32,angle_order=96)),('cutoff108',dict(cutoff=108.,energy_order=12,inner_order=24,angle_order=48)),('crossing72_108',dict(method='crossing',outgoing_limit=108.,energy_order=12,inner_order=24,angle_order=48))]:
  row['runs'][name]=integrate(sp,rad,**kwargs);save()
 ref=np.array(row['runs']['paired_ref']['values']);gross=ref[5]+ref[6];heat_scale=max(abs(ref[3]),gross*sp.T*1e-8);entropy_scale=max(abs(ref[4]),gross*1e-8)
 for name in ('coarse','energy_only','angle_only','direct','cutoff108'):
  val=np.array(row['runs'][name]['values'])
  # LTE moment ratios are ill-conditioned; retain gross-scaled residuals instead.
  for idx,scale in [(1,heat_scale),(2,max(abs(ref[2]),gross*1e-8)),(3,heat_scale),(4,entropy_scale)]:gate(label+' '+name+' '+row['runs'][name]['labels'][idx],abs(val[idx]-ref[idx])/scale,3e-5)
  if name=='direct':
   gate(label+' direct integrated number conservation',abs(val[0])/gross,3e-6)
   gate(label+' direct bath/radiation energy conservation',abs(val[1]+val[3])/max(gross*sp.T,1e-300),3e-12)
   gate(label+' direct radiation+bath entropy identity',abs(val[2]+val[3]/sp.T-val[4])/max(gross,1e-300),3e-12)
  if name!='coarse':gate(label+' '+name+' total entropy nonnegative',max(-val[4],0)/max(gross,1e-300),3e-10)
 if label=='hot':gate('hot radiation heats bath',float(ref[3]<=0),.1)
 if label=='cold':gate('cold radiation cools bath',float(ref[3]>=0),.1)
 cross=np.array(row['runs']['crossing72_108']['values']);gate(label+' boundary number leakage/gross',abs(cross[0])/gross,3e-5);gate(label+' boundary heat correction',abs(cross[3])/max(gross*sp.T,1e-300),3e-5)
 save()
 if time.monotonic()-start>105:raise TimeoutError('Whole study105s limit')
print('checks',sum(c['passed']for c in checks),'/',len(checks),'runtime',time.monotonic()-start)
for c in checks:
 if not c['passed']:print('FAIL',c)
