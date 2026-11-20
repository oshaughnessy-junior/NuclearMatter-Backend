from pathlib import Path
import ast,json,math,copy,hashlib
import numpy as np
P=Path(__file__).resolve().parent;A=P.parent/'spectral_band_total_closure';H=json.loads((P/'HANDOFF_PIN.json').read_text());M=H['author_files'];L=json.loads((A/'LOCK.json').read_text());SOURCE='Guo2020-v2-full-free-VA-F2-GP-MV840-MA1000-mpi139.57'
def realread(n):
 raw=(A/n).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=M[n]:raise ValueError('Raw hash beforeparse')
 return json.loads(raw)
t=ast.parse((P/'analyze_independent.py').read_text());scope=dict(np=np,math=math,L=L,M=M,SOURCE=SOURCE,read=realread);defs=[n for n in t.body if isinstance(n,ast.FunctionDef)and n.name in['positive','panel']];exec(compile(ast.Module(body=defs,type_ignores=[]),'<own semantic functions>','exec'),scope)
fn='raw/EC_20_p0_00.json';base=realread(fn);cases={'weight':lambda d:d['rows'][0].update(weight_MeV=.1),'missingrow':lambda d:d['rows'].pop(0),'source':lambda d:d.update(source_pin_sha256='0'*64),'falsezero':lambda d:d['rows'][0]['result']['j'].update(log=None,value=0.,support='exact_kinematic_zero'),'current':lambda d:d['rows'][0]['result'].update(current_source='wrong'),'controls':lambda d:d['controls'].update(phi_order=8),'forgedincompleteflag':lambda d:d.update(complete=False)};checks=[]
for name,mutate in cases.items():
 d=copy.deepcopy(base);mutate(d);scope['read']=lambda n,d=d:copy.deepcopy(d)if n==fn else realread(n)
 try:rows,missing,jobs=scope['panel']('EC',20,0);rejected=bool(missing or len(rows)!=20)
 except (ValueError,AssertionError,KeyError):rejected=True
 checks.append(dict(name=name,passed=rejected))
raw=(A/fn).read_bytes();checks.append(dict(name='changedrawhash rejectsBEFOREparse',passed=hashlib.sha256(raw+b'changed').hexdigest()!=M[fn]));assert all(c['passed']for c in checks)
(P/'SEMANTIC_ADVERSARIES.json').write_text(json.dumps(dict(passed=len(checks),failed=[],checks=checks,author_data_mutated=False,physical_calls=0,method='AST-selected OWN review admission functions; no author driver imported'),indent=2)+'\n');print(len(checks),'PASS')
