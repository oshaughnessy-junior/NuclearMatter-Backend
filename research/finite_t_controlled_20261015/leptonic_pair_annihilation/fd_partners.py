"""Both partner blocking/direct inverse, reciprocal spinors and entropy ledgers."""
from pathlib import Path
import json,math
import pair as P
H=Path(__file__).parent;checks=[];rows=[]
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
def save(stage):
 (H/'fd_results.json').write_text(json.dumps(dict(stage=stage,checks=checks,rows=rows,all_pass=all(c['passed']for c in checks)),indent=2)+'\n')
for flavor in ['e','mu','tau']:
 for E,c in [(.55,-.7),(2.,.2),(10.,.8)]:
  event=P.cm(4*E*E,c);C=P.Couplings(flavor);scalar=P.amplitude(*event,C)[0]
  for method in ['spinor','inverse_spinor']:gate('fresh reciprocal '+flavor+str(E)+method,P.amplitude(*event,C,method=method)[0]/scalar-1,3e-10)
mu=.17446333200826397;C=P.Couplings('e')
for label,ma in [('LTE',-.5),('forward',-2.1),('inverse',1.1)]:
 rad=P.Radiation(2.,.5,ma);results=[]
 for order,ang in [(16,8),(24,12)]:
  r=P.thermal(2.,mu,C,radial=order,incident=ang,outgoing=ang,azimuth=12,tail=50.,radiation=rad)
  rows.append(dict(label=label,result=r));results.append(r);save(label+str(order))
  scale=r['forward']['number']['value']+r['backward']['number']['value'];A=-(rad.mu_nu+rad.mu_antinu)/2
  gate('local affinity '+label+str(order),r['max_affinity_error'],2e-11)
  gate('positive entropy '+label+str(order),min(0,r['entropy']),0)
  gate('direct entropy '+label+str(order),(r['entropy']-A*r['net']['number'])/scale,3e-12)
  q=r['net']['nu_energy']+r['net']['antinu_energy'];qm=r['net']['electron_energy']+r['net']['positron_energy'];escale=r['forward']['nu_energy']['value']+r['forward']['antinu_energy']['value']+r['backward']['nu_energy']['value']+r['backward']['antinu_energy']['value']
  gate('material/radiation energy '+label+str(order),(q-qm)/escale,3e-12)
  # Both charged species decrease byR: charge and netLe unchanged; each
  # neutrino/antineutrino count increases byR, giving zero radiationLe.
  if label=='LTE':gate('LTE signed net '+str(order),r['net']['number']/scale,3e-12)
  else:gate('net sign '+label+str(order),int((r['net']['number']>0)!=(label=='forward')),0)
 for branch in ['forward','backward']:
  for name in results[0][branch]:gate('FD refinement '+label+branch+name,results[0][branch][name]['value']/results[1][branch][name]['value']-1,3e-6)
 # Independent matter/radiation reservoir entropy from energies and chemical
 # terms: mu_e+mu_e+=0, so only the two neutrino chemical potentials remain.
 r=results[-1];R=r['net']['number'];Q=r['net']['nu_energy']+r['net']['antinu_energy'];sm=-Q/2;sr=(Q-(rad.mu_nu+rad.mu_antinu)*R)/2
 rows.append(dict(entropy_decomposition=dict(label=label,matter=sm,radiation=sr,total=sm+sr)))
 gate('reservoir entropy '+label,(sm+sr-r['entropy'])/(r['forward']['number']['value']+r['backward']['number']['value']),3e-12)
# Empty source is unchanged bitwise for prior valid controls after FD extension.
old=json.loads((H/'pre_fd_thermal_results.json').read_text());previous=next(x['result']for x in old['rows']if x.get('label')=='e'and x['result']['controls']['radial']==12)
new=P.thermal(2.,mu,C,radial=12,incident=6,outgoing=6,azimuth=8,tail=40.)
for name in previous['totals']:gate('empty exact source replay '+name,new['totals'][name]['value']/previous['totals'][name]['value']-1,0.)
rows.append(dict(empty_replay=new))
for n,fn in [('mixedT',lambda:P.thermal(2.,mu,C,radiation=P.Radiation(3,0,0))),('booleanorder',lambda:P.thermal(2.,mu,C,radial=True))]:
 try:fn();rejected=False
 except ValueError:rejected=True
 gate('reject '+n,int(not rejected),0)
gate('positive log underflow flag',int(P.represented(-1000.)['representable']),0)
gate('positive log underflow not physicalzero',int(P.represented(-1000.)['value']is not None),0)
save('complete');print(json.dumps(dict(checks=len(checks),failed=[c for c in checks if not c['passed']])))
