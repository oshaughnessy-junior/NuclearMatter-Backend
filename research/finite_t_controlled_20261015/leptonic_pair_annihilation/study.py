"""Actual matched pair-rich empty-radiation moments and absolute FD massless gate."""
from pathlib import Path
import json,math,sys
from scipy.special import zeta
import pair as P
H=Path(__file__).parent;checks=[];rows=[]
sys.path.insert(0,str(P.R/'relativistic_crossed_thermal'));import kernel
b,metadata=kernel.matched_npe(2.,1e-7,.3)
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and abs(error)<=tol)))
def save(stage):
 (H/'thermal_results.json').write_text(json.dumps(dict(stage=stage,checks=checks,rows=rows,all_pass=all(c['passed']for c in checks)),indent=2)+'\n')
rows.append(dict(matched_state=metadata,mu_e=b.mu_e,radiation='empty nu AND antinu'))
for flavor in ['e','mu']:
 results=[]
 for radial,angular in [(8,4),(12,6),(20,8)]:
  r=P.thermal(2.,b.mu_e,P.Couplings(flavor),radial=radial,incident=angular,outgoing=angular,azimuth=8,tail=40.)
  results.append(r);rows.append(dict(label=flavor,result=r));save(flavor+str(radial))
  gate('actual electron density '+flavor+str(radial),r['electron_density']/metadata['electron']['particle_density']-1,3e-8)
  gate('actual positron density '+flavor+str(radial),r['positron_density']/metadata['electron']['antiparticle_density']-1,3e-8)
  gate('independent angular-integrated number '+flavor+str(radial),r['totals']['number']['value']/r['anchor_number']-1,3e-11)
  q=r['totals']['nu_energy']['value']+r['totals']['antinu_energy']['value']
  gate('independent angular-integrated total energy '+flavor+str(radial),q/r['anchor_total_energy']-1,3e-11)
  mat=r['totals']['electron_energy']['value']+r['totals']['positron_energy']['value']
  gate('separate material/radiation energy '+flavor+str(radial),q/mat-1,3e-12)
 for i,r in enumerate(results[:-1]):
  for name in r['totals']:gate('refinement '+flavor+str(i)+' '+name,r['totals'][name]['value']/results[-1]['totals'][name]['value']-1,3e-6)
 # Isolated tail at identical physical and quadrature controls.
 r=P.thermal(2.,b.mu_e,P.Couplings(flavor),radial=20,incident=8,outgoing=8,azimuth=8,tail=60.)
 rows.append(dict(label=flavor+'_tail60',result=r));save(flavor+'tail')
 for name in r['totals']:gate('isolated tail '+flavor+' '+name,r['totals'][name]['value']/results[-1]['totals'][name]['value']-1,3e-6)
for T in [1.,2.]:
 C=P.Couplings('e');r=P.thermal(T,0.,C,mass=0.,radial=20,incident=6,outgoing=6,azimuth=8,tail=50.)
 rows.append(dict(label='masslessFD'+str(T),result=r));save('massless'+str(T))
 g=C.cv*C.cv+C.ca*C.ca;convert=1/(P.HBAR*P.HC**3)
 rex=49*math.pi**3/64800*P.GF**2*g*T**8*convert
 qex=7*zeta(5,1)/(12*math.pi)*P.GF**2*g*T**9*convert
 gate('massless FD absolute eventT8 '+str(T),r['totals']['number']['value']/rex-1,3e-9)
 gate('massless FD absolute energyT9 '+str(T),(r['totals']['nu_energy']['value']+r['totals']['antinu_energy']['value'])/qex-1,3e-9)
 gate('massless CP flavor-energy symmetry '+str(T),r['totals']['nu_energy']['value']/r['totals']['antinu_energy']['value']-1,3e-12)
save('complete');print(json.dumps(dict(checks=len(checks),failed=[c for c in checks if not c['passed']])))
