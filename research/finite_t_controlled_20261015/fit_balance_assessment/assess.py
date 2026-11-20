"""Research-only assessment of a fitted-source cancellation, not a net-rate API."""
from pathlib import Path
import sys,json,math,hashlib,importlib.util,time
from scipy.optimize import brentq
P=Path(__file__).resolve().parent;ROOT=P.parent;sys.dont_write_bytecode=True

def load(name,path):
 s=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
F=load('fit_balance_optional_positive_API','free_capture_moment_fit/approximation.py');G=load('fit_balance_actual_three_sources','transparent_balance_grid/grid.py');LOCK=json.loads((P/'LOCK.json').read_text());LOCKSHA=hashlib.sha256((P/'LOCK.json').read_bytes()).hexdigest();PINS=dict(LOCK['pins']);PINS.update(json.loads((ROOT/'free_capture_moment_fit/SOURCE_PIN.json').read_text()));G.require_acceptance()
def verify():
 if hashlib.sha256((P/'LOCK.json').read_bytes()).hexdigest()!=LOCKSHA:raise RuntimeError('Locked assessment changed')
 for n,h in PINS.items():
  if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=h:raise RuntimeError('Frozen physical/model/review input changed '+n)

def material_energy(s):return sum(s[k]['energy_density']for k in ['neutron','proton','electron'])
def eos_uY(b,s):
 # Rest-inclusive fixed-T composition response from accepted FD free energies.
 term=lambda name:b.__dict__['mu_'+name[0]]-b.T*s[name]['dmu_dtemperature_fixed_density']
 return term('proton')-term('neutron')+b.mu_e-b.T*s['electron']['dmu_dtemperature_fixed_net_density']

def run():
 if(P/'results.json').exists():raise RuntimeError('Immutable assessment exists')
 verify();start=time.monotonic();numeric=[];model=[];rows=[];gate=LOCK['gates'];sourcecalls=0
 def ck(name,error,tolerance):numeric.append(dict(name=name,error=float(error),gate=tolerance,passed=bool(math.isfinite(error)and abs(error)<=tolerance)))
 def mc(name,error,tolerance):model.append(dict(name=name,error=float(error),gate=tolerance,passed=bool(math.isfinite(error)and abs(error)<=tolerance)))
 def checkpoint():
  (P/'partial.json').write_text(json.dumps(dict(rows=rows,numerical_checks=numeric,model_checks=model,combined_full_source_calls=sourcecalls,seconds=time.monotonic()-start),indent=2)+'\n')
 outside=[]
 for state in LOCK['outside_domain']:
  try:F.approximate_moments(**state);rejected=False
  except ValueError:rejected=True
  ck('reject original outside-fit root '+str(state),0 if rejected else 1,0);outside.append(dict(state=state,rejected=rejected))
 for index,item in enumerate(LOCK['states']):
  T,n=item['T'],item['nB'];reference=json.loads((ROOT/item['reference_file']).read_text());refYe=reference['kinetic_Ye'];trials=[]
  def approximate(y):
   if time.monotonic()-start>108:raise TimeoutError('Assessment108s before beta trial')
   verify();mom=F.approximate_moments(T,n,float(y));b,st=G.EC.matched(T,n,float(y));beta=G.BETA.integrate(G.BETA.K.Bath(**vars(b)),rest_order=24,parent_order=28,orient_order=10,phi_order=16,coordinate='electron',parent_coordinate='direct_p',tail=80.,scenarios=[dict(name='empty',mode='empty')]);bb=beta['scenarios']['empty']['net'];v=mom['moments'];rates=dict(EC=v['R_EC'],PC=v['R_PC'],beta=bb['number']);energy=dict(EC=v['Q_EC'],PC=v['Q_PC'],beta=bb['neutrino_energy']);net=rates['beta']+rates['PC']-rates['EC'];gross=sum(rates.values());Q=sum(energy.values());uY=eos_uY(b,st)
   ck(f'{index} actual beta parent FD {len(trials)}',beta['parent_density']/(n*(1-y))-1,gate['target_density']);ck(f'{index} actual fresh electron mu {len(trials)}',mom['state']['mu_e']-b.mu_e,2e-12)
   row=dict(Ye=y,bath=vars(b),state=st,fit=mom,actual_beta=beta,rates=rates,energy=energy,net=net,gross=gross,dYe_dt=net/n,Qescape=Q,uY_mev=uY,fixed_T_thermostat_power=Q+uY*net)
   trials.append(row);return row
  def fun(y):return approximate(y)['net']
  root=brentq(fun,*item['bracket'],xtol=1e-12,rtol=1e-12);candidate=approximate(root);ck(f'{index} approximate root numerical net/gross',candidate['net']/candidate['gross'],gate['approximate_root_gross_residual']);ck(f'{index} real pair EOS eta_e admission',max(0.,abs(candidate['bath']['mu_e']/T)-3),0)
  actual=[]
  for refined in [False,True]:
   if time.monotonic()-start>108:raise TimeoutError('Assessment108s before combined physical source')
   actual.append(G.evaluate(T,n,root,refined));sourcecalls+=1;checkpoint()
  a,b=actual
  for channel in b['channels']:
   for name,v in b['channels'][channel].items():ck(f'{index} detailed moment refine {channel}/{name}',a['channels'][channel][name]/v-1,gate['individual_detailed_refinement'])
  for profile,r in enumerate(actual):
   for name,e in r['root_residuals'].items():ck(f'{index} profile{profile} EOS {name}',e,gate['EOS_root'])
   ck(f'{index} profile{profile} charge',r['state']['electron']['net_density']/(n*root)-1,gate['EOS_root'])
   for name,e in r['density_integral_residuals'].items():ck(f'{index} profile{profile} FDdensity {name}',e,gate['target_density'])
   ck(f'{index} profile{profile} signed species energy',r['energy_ledger'],3e-10);ck(f'{index} profile{profile} direct energy',r['direct_energy_ledger'],3e-10);ck(f'{index} profile{profile} total electronLe',r['lepton_ledger']/sum(r['rates'].values()),1e-14)
  detailed_gross=sum(b['rates'].values());drift=b['net_material_number_source']/detailed_gross;delta=root-refYe;mc(f'{index} model |deltaYe|',delta,gate['model_abs_deltaYe']);mc(f'{index} model actual detailed net/gross',drift,gate['model_detailed_net_over_gross'])
  per_output={}
  for channel,number,energy in [('EC','R_EC','Q_EC'),('PC','R_PC','Q_PC')]:
   per_output[number]=candidate['rates'][channel]/b['channels'][channel]['number']-1;per_output[energy]=candidate['energy'][channel]/b['channels'][channel]['neutrino_energy']-1
  # Accepted thermodynamic identity independently checked by FD at the forward state.
  h=1e-4;lo=G.EC.matched(T,n,root-h)[1];hi=G.EC.matched(T,n,root+h)[1];uYfd=(material_energy(hi)-material_energy(lo))/(2*h*n);ck(f'{index} EOScomposition energy identity',uYfd-candidate['uY_mev'],2e-5)
  row=dict(T=T,nB=n,reference_root=refYe,candidate_root=root,delta_Ye=delta,actual_detailed_net_over_gross=drift,actual_detailed_dYe_dt=b['dYe_dt'],fit_individual_errors_at_candidate=per_output,fit_beta_candidate=candidate,detailed_both_profiles=actual,actual_fixed_T_thermostat_power=b['escaping_energy']+candidate['uY_mev']*b['net_material_number_source'],approximate_trials=trials,interpretation='Both baths fixed T/nB, empty incident radiation and rapid EM pair equilibrium. Nonzero detailed residual at approximate root is retained; not an exact physical steady state.')
  rows.append(row);checkpoint()
 verify();out=dict(status='AUTHOR_ASSESSMENT_DIFFERENT_REVIEW_REQUIRED',numerical_checks=numeric,numerical_passed=sum(c['passed']for c in numeric),numerical_failed=[c for c in numeric if not c['passed']],model_checks=model,model_passed=sum(c['passed']for c in model),model_failed=[c for c in model if not c['passed']],rows=rows,outside_domain=outside,combined_full_source_calls=sourcecalls,pins=PINS,lock_sha=LOCKSHA,seconds=time.monotonic()-start);(P/'results.json').write_text(json.dumps(out,indent=2)+'\n');print(dict(numerical_passed=out['numerical_passed'],numerical_failed=out['numerical_failed'],model_passed=out['model_passed'],model_failed=out['model_failed'],seconds=out['seconds'],roots=[dict(T=r['T'],nB=r['nB'],Ye=r['candidate_root'],delta=r['delta_Ye'],actual_net_over_gross=r['actual_detailed_net_over_gross'])for r in rows]))
if __name__=='__main__':run()
