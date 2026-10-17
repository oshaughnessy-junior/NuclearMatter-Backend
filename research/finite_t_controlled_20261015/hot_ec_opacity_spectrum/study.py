"""Fixed energies with distinct direct j/chi, refinement and occupation checks."""
from pathlib import Path
import importlib.util,sys,json,math
from scipy.special import expit
P=Path(__file__).parent;sp=importlib.util.spec_from_file_location('hot_EC_spectral_study_functions',P/'spectrum.py');S=importlib.util.module_from_spec(sp);sys.modules[sp.name]=S;sp.loader.exec_module(S)
def main():
 if(P/'results.json').exists():raise RuntimeError('Frozen selected spectrum exists')
 plan=json.loads((P/'EXPERIMENT_PLAN.json').read_text());rows=[];checks=[]
 def ck(n,er,g):checks.append(dict(name=n,error=float(er),gate=g,passed=bool(math.isfinite(er)and abs(er)<=g)))
 for enu in plan['selected_energies_MeV']:
  r=[]
  for controls in plan['selected_rules']:
   n,c,e,ph,t,ot=controls;row=S.integrate(float(enu),lepton_order=n,angle_order=c,energy_order=e,phi_order=ph,tail=t,outer_tail=ot);r.append(row);print(enu,controls,row['seconds'],flush=True)
   ck('separately direct KMS '+str((enu,controls)),row['log_kms_actual']-row['log_kms_expected'],3e-12)
  for i in[0,1]:
   for k in['j','chi']:ck('rule'+str(i)+'to'+str(i+1)+' '+str((enu,k)),r[i][k]['value']/r[i+1][k]['value']-1,3e-6)
  finest=r[-1];j=finest['j']['value'];chi=finest['chi']['value'];mu=finest['equilibrium_mu_nue'];feq=expit((mu-enu)/2);ck('LTE occupation source '+str(enu),(j*(1-feq)-chi*feq)/(j+chi),3e-12);ck('truechi versus relaxation '+str(enu),0 if finest['relaxation']==j+chi and finest['relaxation']>chi else 1,0)
  rows.append(dict(Enu=enu,rules=r,LTE_f=feq,emissivity_cmminus1=j/2.99792458e10,absorption_cmminus1=chi/2.99792458e10));(P/'partial_results.json').write_text(json.dumps(dict(rows=rows,checks=checks),indent=2))
 out=dict(rows=rows,checks=checks,passed=sum(v['passed']for v in checks),failed=[v for v in checks if not v['passed']]);(P/'results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(passed=out['passed'],failed=out['failed']),indent=2))
if __name__=='__main__':main()
