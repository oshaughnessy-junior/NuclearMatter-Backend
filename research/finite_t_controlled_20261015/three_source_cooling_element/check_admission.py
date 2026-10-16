"""Selected admission guards, derivative-bookkeeping adversaries and valid replay."""
from pathlib import Path
import json,importlib.util,sys,math,copy,ast
P=Path(__file__).parent;sp=importlib.util.spec_from_file_location('three_source_admission_client',P/'client.py');M=importlib.util.module_from_spec(sp);sys.modules[sp.name]=M;sp.loader.exec_module(M);C=M.Client();checks=[]
def ck(n,b):checks.append(dict(name=n,passed=bool(b)))
def reject(n,fn):
 try:fn();b=False
 except(ValueError,RuntimeError):b=True
 ck(n,b)
for kw in[dict(T=True,Ye=.3),dict(T=float('nan'),Ye=.3),dict(T=1.79,Ye=.3),dict(T=2.,Ye=.41),dict(T=2.,Ye=True),dict(T=2.,Ye=.3,photons=1),dict(T=2.,Ye=.3,profile='unknown'),dict(T=2.,Ye=.3,action='hydro')]:reject('bad request '+str(kw),lambda kw=kw:C.evaluate(**kw))
for field in['accepted_balance','files','expected_loaded_python']:
 z=copy.deepcopy(C.pin)
 if field=='accepted_balance':z[field]=1
 elif field=='files':z[field].pop('review_transparent_capture_balance/CHECKPOINT.json')
 else:z[field].pop()
 reject('mandatory pin admission '+field,lambda z=z:M.W.validate_pin_record(z))
old=json.loads((P/'pre_admission_guard/endpoint_results.json').read_text());points=[(2.,.3,old['fresh_initial']), (old['endpoint']['base']['state']['T'],old['endpoint']['base']['state']['Ye'],old['endpoint']['base'])];new=[]
def strip(x):
 if isinstance(x,dict):return{k:strip(v)for k,v in x.items()if k not in['seconds','elapsed']}
 if isinstance(x,list):return list(map(strip,x))
 return x
for T,Y,ref in points:
 out=C.evaluate(T,Y,action='source');new.append(out);ck('valid EOS bitwise unchanged '+str((T,Y)),out['state']==ref['state']);ck('valid source all moments bitwise unchanged '+str((T,Y)),strip(out['sources'])==strip(ref['sources']))
q=new[0]['request'];o=new[0]
for name,fn in[('u_dot_sign',lambda a:a['sources'].update(u_dot=-a['sources']['u_dot'])),('Ye_dot_sign',lambda a:a['sources'].update(Ye_dot=-a['sources']['Ye_dot'])),('photon_doublecount',lambda a:a['state']['photon'].update(energy_density=2*a['state']['photon']['energy_density'])),('Cv_bool',lambda a:a['state'].update(cv=True)),('moment_falsezero',lambda a:a['sources']['channels']['beta'].update(number=0)),('positron_sign',lambda a:a['sources']['number_sources'].update(positron=-a['sources']['number_sources']['positron'])),('identity',lambda a:a.update(identity='0'*64)),('missingloaded',lambda a:a['loaded_sources'].pop('hot_electron_capture_moments/electron_capture.py')),('control',lambda a:a['sources']['controls'].update(EC=[4,4,4,30.]))]:
 a=copy.deepcopy(o);fn(a);reject('reject synthetic '+name,lambda a=a:C.validate(a,q))
oldtree=ast.parse((P/'pre_admission_guard/worker.py').read_text());newtree=ast.parse((P/'worker.py').read_text())
for name in['state','source']:
 get=lambda tr:ast.dump(next(n for n in tr.body if isinstance(n,ast.FunctionDef)and n.name==name),include_attributes=False)
 ck('unchanged physical function AST '+name,get(oldtree)==get(newtree))
ck('physical controls unchanged',M.W.CONTROLS==dict(base=dict(EC=[20,32,8,50.],PC=[28,44,10,60.],beta=[16,20,8,12,'pair','laguerre',60.]),fine=dict(EC=[28,48,12,60.],PC=[32,52,12,60.],beta=[24,28,10,16,'electron','direct_p',80.]),tail=dict(EC=[28,48,12,80.],PC=[32,52,12,80.],beta=[24,28,10,16,'electron','direct_p',100.])))
out=dict(checks=checks,passed=sum(z['passed']for z in checks),failed=[z for z in checks if not z['passed']],fresh_replays=new,client_calls=C.calls,physical_trajectories_rerun=False);(P/'admission_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(passed=out['passed'],failed=out['failed']),indent=2))
