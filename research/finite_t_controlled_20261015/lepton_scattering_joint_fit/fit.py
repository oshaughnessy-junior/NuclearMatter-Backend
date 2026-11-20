"""Training-only joint moment selection; immutable prevalidation lock."""
from pathlib import Path
import json,math,hashlib
import numpy as np
from numpy.polynomial.chebyshev import chebfit,chebval
P=Path(__file__).parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 if(P/'FITTING_LOCK.json').exists()or(P/'validation.json').exists():raise RuntimeError('Model already locked or validation exists')
 reg=json.loads((P/'PREREGISTRATION.json').read_text());data=json.loads((P/'training.json').read_text())
 if data['failed']or data['preregistration_sha']!=sha(P/'PREREGISTRATION.json')or [r['E']for r in data['rows']]!=reg['training_MeV']:raise ValueError('Qualified exact training required')
 E=np.array(reg['training_MeV']);x=2*np.log(E)/math.log(30)-1;total=[z['fine']['total']for z in data['rows']];models={}
 for k in['rate','transport_rate','g']:
  y=np.array([r['neutrino_energy_change']/(2*r['rate'])if k=='g'else math.log(r[k])for r in total]);candidates=[]
  for degree in reg['candidate_degrees']:
   c=chebfit(x,y,degree);tr=chebval(x,c)-y;errors=[]
   for i in range(len(x)):
    mask=np.arange(len(x))!=i;ci=chebfit(x[mask],y[mask],degree);errors.append(float(chebval(x[i],ci)-y[i]))
   if k!='g':tr=np.expm1(tr);errors=np.expm1(errors)
   trmax=float(max(abs(tr)));loomax=float(max(abs(np.array(errors))));tg=.0005 if k=='g'else.002;lg=.002 if k=='g'else.005
   candidates.append(dict(degree=degree,coefficients=c.tolist(),train_max=trmax,LOO_max=loomax,LOO_errors=list(map(float,errors)),passed=trmax<=tg and loomax<=lg))
  selected=next((c for c in candidates if c['passed']),min(candidates,key=lambda c:c['LOO_max']));models[k]=dict(selected=selected,candidates=candidates)
 xx=np.linspace(-1,1,4001);ee=np.exp((xx+1)*math.log(30)/2);rr=np.exp(chebval(xx,models['rate']['selected']['coefficients']));rt=np.exp(chebval(xx,models['transport_rate']['selected']['coefficients']));g=chebval(xx,models['g']['selected']['coefficients']);bounds=bool(np.all(ee+2*g>0)and np.all(rt<=2*rr))
 m=dict(version=reg['version'],state=reg['state'],domain_MeV=reg['domain_MeV'],models=models,x='2log(E)/log30-1',representation=reg['representation'],fallback=reg['fallback'],units={'rate':'s^-1 per incidentnu_e','transport_rate':'s^-1 per incidentnu_e','outgoing_energy_moment':'MeV s^-1 per incidentnu_e','neutrino_energy_change':'MeV s^-1 per incidentnu_e'},training_sha=sha(P/'training.json'),preregistration_sha=sha(P/'PREREGISTRATION.json'),source_pins=data['rows'][0]['fine']['pins'],dense_physical_bounds=bounds,dense_bounds_points=4001,qualification=reg['scope'])
 (P/'model.json').write_text(json.dumps(m,indent=2)+'\n');lock=dict(model_sha=sha(P/'model.json'),training_sha=m['training_sha'],preregistration_sha=m['preregistration_sha'],fresh_validation_file_absent=not(P/'validation.json').exists(),selection_passed=all(z['selected']['passed']for z in models.values()),dense_bounds=bounds)
 (P/'FITTING_LOCK.json').write_text(json.dumps(lock,indent=2)+'\n');print(json.dumps(dict(lock=lock,selected={k:(v['selected']['degree'],v['selected']['train_max'],v['selected']['LOO_max'])for k,v in models.items()}),indent=2))
if __name__=='__main__':main()
