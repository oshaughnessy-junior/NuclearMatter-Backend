"""Hot phase/orientation/parent/tail controls and conditional-spectrum gate."""
from pathlib import Path
import json,math
import hot_moments as M
from conditional import conditional
from hot_spectrum import spectral
H=Path(__file__).parent;initial=json.loads((H/'results.json').read_text());reference=initial['rows'][-1]['result'];b=M.K.Bath(**reference['bath']);sc=[z['scenario']for z in reference['scenarios'].values()];rows=[];checks=[]
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and abs(error)<=tol)))
def save(phase):
 (H/'controls.json').write_text(json.dumps(dict(phase=phase,checks=checks,rows=rows,all_pass=all(c['passed']for c in checks)),indent=2)+'\n')
base=dict(rest_order=32,parent_order=24,orient_order=8,phi_order=12,scenarios=sc)
for name,change in [('alternate_electron_phase',dict(coordinate='electron')),('direct_parent_p',dict(parent_coordinate='direct_p',tail=50.)),('direct_parent_tail',dict(parent_coordinate='direct_p',tail=80.)),('low_orientation',dict(orient_order=4,phi_order=4))]:
 c=dict(base);c.update(change);r=M.integrate(b,**c);rows.append(dict(label=name,result=r));save(name)
 gate(name+' actual parent density',r['parent_density']/7e-8-1,3e-10)
 for label,s in r['scenarios'].items():
  for branch in ['forward','backward']:
   for key in M.NAMES:
    a=s[branch][key]['value'];z=reference['scenarios'][label][branch][key]['value']
    if z:gate(name+' '+label+' '+branch+' '+key,a/z-1,3e-6)
# Original old low-T integrated API remains rejected at this hot state.
try:M.K.spectral(b,.42);rejected=False
except ValueError:rejected=True
gate('old lowT spectral guard remains intact',int(not rejected),0)
cond=[]
for order,parent,phi in [(12,16,8),(20,24,12)]:
 r=conditional(b,E_lab=.42,order=order,parent_order=parent,phi_order=phi);cond.append(r);rows.append(dict(conditional=r));save('conditional '+str(order))
tensor=[]
for angle,lepton,inner,widths in [(10,12,16,9.),(16,20,24,13.)]:
 r=spectral(b,.42,angle_order=angle,lepton_order=lepton,energy_order=inner,widths=widths,tail=70.)
 tensor.append(r);rows.append(dict(new_hot_reference=r));save('hot reference '+str(lepton))
for key in ['j','chi']:
 gate('new conditional refinement '+key,cond[0][key]['value']/cond[1][key]['value']-1,3e-6)
 gate('new hot tensor reference refinement '+key,tensor[0][key]['value']/tensor[1][key]['value']-1,3e-6)
 gate('new conditional/tensor absolute '+key,cond[-1][key]['value']/tensor[-1][key]['value']-1,3e-6)
save('complete');print(json.dumps(dict(checks=len(checks),failed=[c for c in checks if not c['passed']])))
