"""Bounded physical free npe positron capture pilot; raw checkpoints."""
from pathlib import Path
import sys, json, time, math
import pc_kernel as pc
sys.path.insert(0,str(pc.R/'relativistic_capture_oracle'));import events as ev
H=Path(__file__).parent;rows=[];checks=[]
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and abs(error)<=tol)))
def save(phase):
 (H/'results.json').write_text(json.dumps(dict(phase=phase,checks=checks,rows=rows,all_pass=all(x['passed']for x in checks)),indent=2)+'\n')
b,metadata=pc.matched(2.,1e-7,.3)
rows.append(dict(matched_state=metadata,bath=pc.asdict(b)))
for key,target in [('neutron',7e-8),('proton',3e-8)]:gate('matched '+key+' root',metadata[key]['density']/target-1,3e-10)
gate('matched net electron density',metadata['electron']['net_density']/3e-8-1,3e-10)
threshold=((pc.crossed.MN+pc.ME)**2-pc.crossed.MP**2)/(2*pc.crossed.MP)
rows.append(dict(proton_rest_threshold_MeV=threshold,mu_plus=-b.mu_e,muons='absent',pair_density_ratio=metadata['electron']['antiparticle_density']/metadata['electron']['particle_density']))
eb=ev.Bath(b.T,pc.crossed.MP,pc.crossed.MN,b.mu_p,b.mu_n,-b.mu_e,Enu=5.,ml=pc.ME,anti=True,Mweak=pc.crossed.MW)
for order in [8,12]:
 start=time.monotonic();out=ev.integrate(eb,radial=order,incident=order,outgoing=2*order,azimuth=order,tail=50.)
 rows.append(dict(kind='CM_events',order=order,result=out,runtime=time.monotonic()-start));save('CM events '+str(order))
 gate('CM target density '+str(order),out['density']/3e-8-1,3e-8)
for order in [4,8,12]:
 out=pc.integrate(b,5.,lepton_order=order,angle_order=order,energy_order=12 if order<12 else 20,phi_order=4,tail=50.,outer_tail=24.)
 rows.append(dict(kind='response',order=order,result=out));save('response '+str(order))
 gate('response direct KMS '+str(order),out['log_kms_actual']-out['log_kms_expected'],3e-11)
es=[r['result']for r in rows if r.get('kind')=='CM_events'];rs=[r['result']for r in rows if r.get('kind')=='response']
for key in ['j','chi']:
 gate('CM refinement '+key,es[0][key]['value']/es[1][key]['value']-1,3e-6)
 for i in range(len(rs)-1):gate('response refinement '+str(i)+' '+key,rs[i][key]['value']/rs[-1][key]['value']-1,3e-6)
 gate('absolute CM vs response '+key,rs[-1][key]['value']/es[-1][key]['value']-1,3e-6)
save('complete');print(json.dumps(dict(checks=len(checks),failures=[c for c in checks if not c['passed']])))
