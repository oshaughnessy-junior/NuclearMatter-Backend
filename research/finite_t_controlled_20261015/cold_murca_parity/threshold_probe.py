"""Legacy branch bookkeeping vs primary cold polygon, including failed extrapolation."""
from pathlib import Path
import json,math
import run_parity as P
HERE=Path(__file__).resolve().parent;checks=[];rows=[];extrapolation_failures=[]
for n,kfn in enumerate([1.99,2.,2.01]):
 s=dict(T=1e9,kfn=kfn,kfp=.5,kfe=.5,kfm=0.,mstn=.78,mstp=.69,tcn=0.,tcp=0.,i=5,isf=5)
 t=dict(s,mstn=s['mstn']*1.2);r,q=P.original([s,t],'polygon_'+str(n));assert r.returncode==0 and len(q)==2
 # Infer branches from TWO original-code totals, not instrumentation or a port.
 qp=(1.2**3*q[0]-q[1])/(1.2**3-1.2);qn=q[0]-qp;fn,fp=P.normal_branches(s)
 for label,error in [('inferred neutron branch',(qn-fn)/q[0]),('inferred proton branch',(qp-fp)/q[0]),('actual NMB total',P.evaluate(s)/q[0]-1)]:checks.append(dict(name=str(kfn)+' '+label,error=error,tolerance=3e-12,passed=abs(error)<=3e-12))
 slack=s['kfe']+3*s['kfp']-s['kfn'];expected_allowed=slack>=0
 rows.append(dict(state=s,derived_nB=(kfn**3+.5**3)/(3*math.pi**2),derived_Yp=.5**3/(kfn**3+.5**3),polygon_slack=slack,primary_proton_allowed=expected_allowed,original_total=q[0],inferred_neutron=qn,inferred_proton=qp,formula_proton=fp))
 if slack==0:checks.append(dict(name='exact polygon edge proton branch zero',error=abs(qp)/q[0],tolerance=3e-12,passed=abs(qp)/q[0]<=3e-12))
 if slack<0:extrapolation_failures.append(dict(name='primary forbidden-side proton branch must vanish',error=abs(qp)/q[0],tolerance=3e-12,passed=abs(qp)/q[0]<=3e-12,classification='SHARED LEGACY RECIPE extrapolation, not NMB port mismatch; excluded physical scope'))
results=json.loads((HERE/'results.json').read_text());neg=next(x for x in results['excluded_recipe_diagnostics']if x['name']=='negative_alpha_low_kfn')
extrapolation_failures.append(dict(name='nonnegative physical emissivity at low-kfn extrapolation',value=neg['fortran'],passed=neg['fortran']>=0,classification='SHARED LEGACY RECIPE outside fitted alpha domain; excluded physical scope'))
out=dict(checks=checks,passed=sum(x['passed']for x in checks),failed=[x for x in checks if not x['passed']],rows=rows,physical_extrapolation_gates=extrapolation_failures,physical_extrapolation_failures=sum(not x['passed']for x in extrapolation_failures),source='Gusakov2002 Eq14/15 cold proton polygon; no replacement rate implemented')
(HERE/'threshold_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k]for k in ['passed','failed','physical_extrapolation_failures']}))
