"""Training-only compact coefficient selection, locked before heldout calls."""
from pathlib import Path
import json,hashlib,time,math
import numpy as np
from numpy.polynomial.chebyshev import chebfit,chebval
from scipy.integrate import quad
from scipy.special import expit
P=Path(__file__).parent

def main():
 if (P/'FITTING_LOCK.json').exists():raise RuntimeError('Refusing to overwrite a locked training fit')
 reg=json.loads((P/'PREREGISTRATION.json').read_text());data=json.loads((P/'training_references.json').read_text())
 if [x['E']for x in data['rows']]!=reg['training_MeV']or data['failed']:raise ValueError('Complete numerically qualified preregistered training references required')
 E=np.array(reg['training_MeV']);x=2*np.log(E)/math.log(30)-1;models={};checks=[];density_checks=[]
 m=.51099895069;T=2.;mu=.17446333200826397
 densities=[quad(lambda p:p*p*expit((sign*mu-math.hypot(p,m))/T),0,250,epsabs=1e-11,epsrel=2e-12)[0]/(math.pi**2*197.3269804**3)for sign in[1,-1]]
 for row in data['rows']:
  for target,ref in zip(row['fine']['targets'],densities):
   e=target['target_density']/ref-1;density_checks.append(dict(E=row['E'],positron=target['positron'],error=e,tolerance=2e-9,passed=abs(e)<=2e-9))
 if not all(z['passed']for z in density_checks):raise ValueError('Actual pair-population forward gates failed')
 for key in reg['positive_observables']:
  y=np.log([z['fine']['total'][key]for z in data['rows']]);candidates=[]
  for degree in reg['fit']['candidate_degrees']:
   c=chebfit(x,y,degree);train=float(np.max(abs(np.exp(chebval(x,c)-y)-1)));errors=[]
   for i in range(len(x)):
    mask=np.arange(len(x))!=i;ci=chebfit(x[mask],y[mask],degree);errors.append(float(math.expm1(chebval(x[i],ci)-y[i])))
   loo=max(abs(e)for e in errors);candidates.append(dict(degree=degree,coefficients=c.tolist(),training_max_relative=train,LOO_max_relative=loo,LOO_relative_errors=errors,passes_training_policy=train<=.003 and loo<=.007))
  chosen=next((z for z in candidates if z['passes_training_policy']),min(candidates,key=lambda z:z['LOO_max_relative']));models[key]=dict(selected=chosen,candidates=candidates);checks.append(dict(name='training-only selection '+key,passed=chosen['passes_training_policy']))
 source_pins=data['rows'][0]['fine']['pins']
 for row in data['rows']:
  if row['fine']['pins']!=source_pins:raise ValueError('Mixed physical source identity')
 model=dict(version=reg['version'],state=reg['state'],domain_MeV=reg['domain_MeV'],x_definition='x=2*log(E/1MeV)/log(30)-1',models=models,units=dict(rate='s^-1 per incidentnue',transport_rate='s^-1 per incidentnue',outgoing_energy_moment='MeV s^-1 per incidentnue',neutrino_energy_change='MeV s^-1 per incidentnue'),weak_parameters=dict(GF_MeV_minus2=1.1663785e-11,me_MeV=m,sin2w=.23129,CV=.96258,CA=.5,Vud='ABSENT'),source_pins=source_pins,preregistration_sha=hashlib.sha256((P/'PREREGISTRATION.json').read_bytes()).hexdigest(),training_reference_sha=hashlib.sha256((P/'training_references.json').read_bytes()).hexdigest(),coefficient_selection='TRAINING ONLY; heldout sources not yet computed',selection_checks=checks,actual_target_densities=densities,density_checks=density_checks,qualification='Optional fixed-state moment fit; heldout report is measured evidence, not a uniform certificate. No spectral/AD-chain substitution.')
 (P/'model.json').write_text(json.dumps(model,indent=2));sha=hashlib.sha256((P/'model.json').read_bytes()).hexdigest();(P/'FITTING_LOCK.json').write_text(json.dumps(dict(model_sha=sha,preregistration_sha=model['preregistration_sha'],training_reference_sha=model['training_reference_sha'],heldout_source_file_absent=not(P/'heldout_references.json').exists(),selection_checks=checks),indent=2));print(json.dumps(dict(model_sha=sha,selected={k:v['selected']['degree']for k,v in models.items()},checks=checks),indent=2))
if __name__=='__main__':main()
