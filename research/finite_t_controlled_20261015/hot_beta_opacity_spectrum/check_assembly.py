"""Physicalnode/logadmission includingfalse-empty rejection, no sourcecalc."""
from pathlib import Path
import importlib.util,sys,json,copy
P=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('beta_own_assembly',P/'total.py');m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
b,meta=m.S.matched();c=m.contract('fine');rows=json.loads((P/'total_fine_0.json').read_text())['rows'];checks=[]
for row in rows:
 try:m.validate(c,row,b,meta);ok=True
 except RuntimeError:ok=False
 checks.append(dict(name='actualnode '+str(row['index']),passed=ok))
for label,mutate in [('falseemptyboth',lambda x:x['spectrum'].update(j=m.S.represented(None),chi=m.S.represented(None),relaxation=0.,log_kms_actual=None)),('falseemptyj',lambda x:x['spectrum'].update(j=m.S.represented(None))),('indexbool',lambda x:x.update(index=True)),('weightnan',lambda x:x.update(weight=float('nan'))),('source',lambda x:x['spectrum'].update(implementation_sha256='changed')),('bath',lambda x:x['spectrum']['bath'].update(mu_e=0.)),('lognan',lambda x:x['spectrum']['j'].update(log=float('nan'))),('value',lambda x:x['spectrum']['j'].update(value=7.)),('reason',lambda x:x['spectrum']['j'].update(reason='empty_support')),('controls',lambda x:x['spectrum']['controls'].update(cone_order=100)),('units',lambda x:x['spectrum'].update(units='wrong')),('chemical',lambda x:x['spectrum'].update(equilibrium_mu_antinue=0.))]:
 row=copy.deepcopy(rows[0]);mutate(row)
 try:m.validate(c,row,b,meta);ok=False
 except (RuntimeError,ValueError,ArithmeticError,TypeError):ok=True
 checks.append(dict(name='reject '+label,passed=ok))
out=dict(checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']]);(P/'assembly_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out['passed'],out['failed'])
