"""Recover completed physical trial data after reporter-only NumPybool failure.

Does not replay any spectral integration or the completed fresh total reference.
"""
from pathlib import Path
import importlib.util,sys,json,math
from scipy.special import expit
P=Path(__file__).resolve().parent;sys.dont_write_bytecode=True
sp=importlib.util.spec_from_file_location('hotPC_recover_function',P/'spectrum.py');S=importlib.util.module_from_spec(sp);sys.modules[sp.name]=S;sp.loader.exec_module(S)
d=json.loads((P/'partial_results.json').read_text());rows=d['completed']+[dict(E=d['E'],rules=d['rules'])];checks=[]
for row in rows:
 E=row['E'];vals=row['rules']
 for key in ['j','chi']:
  for i in [0,1]:
   err=vals[i][key]['value']/vals[i+1][key]['value']-1;checks.append(dict(name=f'{E} rule{i}to{i+1} {key}',error=err,tolerance=3e-6,passed=abs(err)<=3e-6,kind='coarse'if i==0 else'qualification'))
 r=vals[-1];eq=r['equilibrium_mu_antinue'];f=float(expit((eq-E)/2));j,chi=r['j']['value'],r['chi']['value'];checks.append(dict(name=f'{E} separate KMS',error=r['log_kms_actual']-r['log_kms_expected'],tolerance=3e-12,passed=abs(r['log_kms_actual']-r['log_kms_expected'])<=3e-12));checks.append(dict(name=f'{E} trueabs vsrelax',passed=chi>0 and r['relaxation']==j+chi and r['relaxation']>chi));err=(j*(1-f)-chi*f)/(j*(1-f)+chi*f);checks.append(dict(name=f'{E} actualLTE',error=err,tolerance=3e-12,passed=abs(err)<=3e-12))
b,_=S.matched();w=S.differential(b,1.7,.5987748936352157,-.1995027501115606,energy_order=24,phi_order=8,tail=80.);checks.append(dict(name='belowrestthreshold actualthermal differential rerun afterreportfailure',passed=w['j']['value']>0 and w['chi']['value']>0));ref=json.loads((P/'FRESH_EVENT_REFERENCE.json').read_text());err=ref['neutron_density']/7e-8-1;checks.append(dict(name='freshCMFDneutrondensity',error=err,tolerance=3e-8,passed=abs(err)<=3e-8))
out=dict(rows=rows,checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],below_rest_threshold=w,seconds=sum(v['seconds']for r in rows for v in r['rules']),recovery='All9actualspectra+freshCMtotal recovered fromcompletedraw; smallbelowthreshold witness rerun. No synthetic source or total replay.')
(P/'results.json').write_text(json.dumps(out,indent=2)+'\n');print({k:out[k]for k in ['passed','failed','seconds']})
