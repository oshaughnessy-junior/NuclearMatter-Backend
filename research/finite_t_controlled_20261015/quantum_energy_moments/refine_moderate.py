"""Targeted actual failed-grid repair; preserve original moderate_results."""
from integrated import *
import json,time
P=Path(__file__).resolve().parent;d=json.loads((P/'moderate_results.json').read_text());m=slab.matter_state(2.,.002,.001);start=time.monotonic()
base=next(r for r in d['runs'] if r['flavor']=='nue' and r['outer_order']==20)
newpanels=[];checks=[]
for lo,hi in [(24.,40.),(40.,70.),(70.,120.)]:
 values=[]
 for order in (32,48):
  xs,ws=leggauss(order);v=np.zeros(len(NAMES))
  for x,w in zip(xs,ws):
   E=float(lo+(x+1)*(hi-lo)/2)
   spectral_values,c=spectral(m,'nue',E,1e5,48,40.)
   v+=float(w*(hi-lo)/2)*spectral_values
  values.append(v)
 for k,a,b in zip(NAMES,*values):
  error=abs(a/b-1);checks.append(dict(name=f'panel{lo}-{hi} outer32->48 {k}',error=float(error),tolerance=2e-5,passed=bool(error<2e-5)))
 newpanels.append(dict(lo=lo,hi=hi,coarse32=values[0].tolist(),fine48=values[1].tolist()))
old=np.array([base['values'][k] for k in NAMES]);refined=old.copy()
for panel in newpanels:
 original=next(p for p in base['panels'] if p['lo']==panel['lo'] and p['hi']==panel['hi'])
 refined+=np.array(panel['fine48'])-np.array(original['values'])
values=dict(zip(NAMES,map(float,refined)))
values['radiation_number_gain_cm2_s']=float(refined[6]-refined[7]);values['radiation_energy_gain_MeV_cm2_s']=float(refined[12]-refined[13]);values['bath_energy_source_erg_cm2_s']=-values['radiation_energy_gain_MeV_cm2_s']*MEV_ERG;values['bath_electron_lepton_source_cm2_s']=-values['radiation_number_gain_cm2_s'];values['emission_energy_erg_cm3_s']=values['emission_energy_MeV_cm3_s']*MEV_ERG
# Verify the unrefined remainder independently from the original ladders.
runs={q['outer_order']:q for q in d['runs'] if q['flavor']=='nue'}
for order in (8,12):
 a=np.sum([q['values'] for q in runs[order]['panels'] if not (24<=q['lo'] and q['hi']<=120)],axis=0)
 b=np.sum([q['values'] for q in runs[20]['panels'] if not (24<=q['lo'] and q['hi']<=120)],axis=0)
 for key,x,y in zip(NAMES,a,b):
  error=abs(x/y-1);checks.append(dict(name=f'remainder outside24-120 outer{order}->20 {key}',error=float(error),tolerance=2e-5,passed=bool(error<2e-5)))
report=dict(passed=all(c['passed'] for c in checks),checks=checks,refined_nue_values=values,panel_refinements=newpanels,original20_to_refined_relative_changes=dict(zip(NAMES,map(float,abs(refined/old-1)))),runtime_seconds=time.monotonic()-start,qualification='Repairs only dominant failed nue bulk panels; remaining panels retain outer20/tail320 and sampled internal48 qualification. Original8/12 grids remain failed.')
(P/'moderate_refined_results.json').write_text(json.dumps(report,indent=2)+'\n');print(sum(c['passed'] for c in checks),'/',len(checks),report['runtime_seconds'])
for c in checks:
 if not c['passed']:print('FAIL',c)
