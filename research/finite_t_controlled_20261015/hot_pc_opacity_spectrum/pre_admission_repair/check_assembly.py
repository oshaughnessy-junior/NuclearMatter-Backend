from pathlib import Path
import importlib.util,sys,json,copy,math
P=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('hotPC_own_assembly_functions',P/'total.py');m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
b,meta=m.S.matched();c=m.contract('fine');rows=json.loads((P/'total_fine_0.json').read_text())['rows'];checks=[]
for i,row in enumerate(rows):
 try:m.validate_row(c,row,b,meta);good=True
 except RuntimeError:good=False
 checks.append(dict(name='actualnode '+str(i),passed=good))
for name,path,value in [('log',['spectrum','j','log'],float('nan')),('ordinary',['spectrum','j','value'],7.),('reason',['spectrum','j','reason'],'empty_support'),('index',['index'],True),('weight',['weight'],float('nan')),('bath',['spectrum','bath','mu_e'],0.),('controls',['spectrum','controls','phi_order'],8),('source',['spectrum','implementation_sha256'],'changed'),('truechi',['spectrum','relaxation'],1.),('chemical',['spectrum','equilibrium_mu_antinue'],0.),('units',['spectrum','units'],'wrong')]:
 row=copy.deepcopy(rows[0]);d=row
 for k in path[:-1]:d=d[k]
 d[path[-1]]=value
 try:m.validate_row(c,row,b,meta);good=False
 except (RuntimeError,ArithmeticError,ValueError):good=True
 checks.append(dict(name='rejecttamper '+name,passed=good))
out=dict(checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']]);(P/'assembly_controls.json').write_text(json.dumps(out,indent=2)+'\n');print(out['passed'],out['failed'])
