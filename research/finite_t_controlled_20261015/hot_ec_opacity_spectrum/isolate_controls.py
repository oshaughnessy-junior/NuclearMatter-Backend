"""Selected E2 inner/outer tail and azimuth variations, fixed other controls."""
from pathlib import Path
import importlib.util,sys,json,math
P=Path(__file__).parent;sp=importlib.util.spec_from_file_location('hot_EC_spectrum_isolated_controls',P/'spectrum.py');S=importlib.util.module_from_spec(sp);sys.modules[sp.name]=S;sp.loader.exec_module(S);base=json.loads((P/'results.json').read_text())['rows'][0]['rules'][-1];checks=[];rows=[]
for name,change in[('inner_tail70to90',dict(tail=90.)),('outer_tail40to50',dict(outer_tail=50.)),('azimuth8to12',dict(phi_order=12))]:
 kw=dict(lepton_order=12,angle_order=20,energy_order=16,phi_order=8,tail=70.,outer_tail=40.);kw.update(change);r=S.integrate(2.,**kw);rows.append(dict(name=name,result=r))
 for k in['j','chi']:
  error=r[k]['value']/base[k]['value']-1;checks.append(dict(name=name+' '+k,error=error,gate=3e-6,passed=bool(abs(error)<=3e-6)))
out=dict(rows=rows,checks=checks,passed=sum(v['passed']for v in checks),failed=[v for v in checks if not v['passed']]);(P/'isolated_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(passed=out['passed'],failed=out['failed']),indent=2))
