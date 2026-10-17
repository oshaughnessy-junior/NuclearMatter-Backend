"""Tail-center scope extension; selected physical coefficients unchanged exactly."""
from pathlib import Path
import sys,json,math
sys.dont_write_bytecode=True
from muon_events import *
H=Path(__file__).parent;old=json.loads((H/'initial_refinement_results.json').read_text());m=old['rows'][0]['EOS'];b=Bath(2.,m['muon']['chemical_potential'],m['electron']['chemical_potential']);rad=Radiation(**old['rows'][0]['radiation']);checks=[];rows=[]
def gate(name,error,tol=0.):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=math.isfinite(error)and abs(error)<=tol))
for label,radiation,blocking in [('unblocked',None,False),('transparent',None,True),('FDpartners',rad,True)]:
 prior=next(x['result']for x in old['rows']if x.get('label')==label)
 r=integrate(b,radiation,electron_blocking=blocking,parent_order=12,pair_order=16,pair_angle=8,orientation_order=6,azimuth_order=8)
 for branch in ['forward','backward']:
  for key in prior[branch]:gate('exact selected '+label+' '+branch+' '+key,float(r[branch][key]!=prior[branch][key]))
 for key in prior['net']:gate('exact selected '+label+' net '+key,r['net'][key]-prior['net'][key])
 rows.append(dict(label=label,upper=r['controls']['parent_kinetic_upper'],qualifies='only equality to previous selected coefficients'))
# Deliberately changed chemical cutoff contract; no rate accuracy assertion.
high=Radiation(2.,60.,60.);expected=max(0.,b.mu_mu-M,b.mu_e+120.-M)+100.
gate('inverse chemical cutoff included',parent_kinetic_upper(b,high,True,50.)-expected)
gate('transparent keeps parent chemical cutoff',parent_kinetic_upper(b,None,True,50.)-(max(0.,b.mu_mu-M)+100.))
(H/'tail_contract_results.json').write_text(json.dumps(dict(passed=all(x['passed']for x in checks),checks=checks,rows=rows,high_chemical_cutoff=expected,high_case_rate_NOT_qualified=True),indent=2)+'\n');print(len(checks),all(x['passed']for x in checks))
