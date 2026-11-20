"""Pinned independent recheck of paired NC source repair; old findings retained."""
from pathlib import Path
import sys,json,hashlib,importlib.util,math,ast
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
NC=ROOT/'neutral_current'
sys.path.insert(0,str(NC))
import free_nc as nc
EXPECTED='b802b1799188dfb929ff44f2094db3e578009865400a5a548595f9f2c48d126c'
assert hashlib.sha256((NC/'free_nc.py').read_bytes()).hexdigest()==EXPECTED
oldfile=NC/'pre_repair_20261015T1250/free_nc.py'
spec=importlib.util.spec_from_file_location('old_nc_review',oldfile)
old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
checks=[]
def add(name,error,tol):
 checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(error<tol)))
def function_ast(path,name):
 return ast.dump(next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name))
for name in ['angular_weight','log_kernel','kernel','bare_static_cross_sections']:
 checks.append(dict(name='unchanged physical function AST '+name,
                    passed=function_ast(oldfile,name)==function_ast(NC/'free_nc.py',name)))
s=nc.state_density(1.3,.0013,offset=17.)
E=np.array([3.5,5.,7.5,11.,18.]);w=np.array([.3,1.3,.8,2.1,4.]);f=np.array([.91,.07,.63,.21,.005])
c=nc.Couplings(.2,.7)
r=nc.discrete_collision(s,E,w,f,c,angle_order=40)
prior=old.discrete_collision(s,E,w,f,c,angle_order=40)
checks.append(dict(name='unchanged transition-rate matrix on original held-out state',passed=bool(np.array_equal(r['transition_rates'],prior['transition_rates']))))
g,A=r['density_weights'],r['transition_rates']
flux=g[:,None]*A*f[:,None]*(1-f[None,:]);gross=flux.sum()
pairentropy=0.;bath=0.
for i in range(len(E)):
 for j in range(i):
  F,B=flux[i,j],flux[j,i]
  pairentropy+=(F-B)*math.log(F/B)
  bath+=(E[i]-E[j])*(F-B)
def entropy(result):return -np.dot(result['number_source'],np.log(f)-np.log1p(-f)+E/s.T)
newerr=abs(entropy(r)/pairentropy-1);olderr=abs(entropy(prior)/pairentropy-1)
add('original strict entropy identity after repair',newerr,2e-12)
add('independent pair bath heat versus returned radiation energy',abs(bath+r['net_neutrino_energy_source'])/max(abs(bath),abs(r['net_neutrino_energy_source'])),2e-12)
add('number conservation after pair accumulation',abs(r['net_number_source'])/gross,1e-13)
checks.append(dict(name='original failure reproduced on preserved source',value=olderr,passed=bool(olderr>2e-12)))
for coupling in [nc.NEUTRON,nc.PROTON]:
 new=nc.moments(s,7.3,coupling,angle_order=24)
 prev=old.moments(s,7.3,coupling,angle_order=24)
 checks.append(dict(name='unchanged held-out valid-control moments '+str(coupling),
                    passed=all(new[k]==prev[k] for k in new)))
assert hashlib.sha256((NC/'free_nc.py').read_bytes()).hexdigest()==EXPECTED
out=dict(source_sha256=EXPECTED,preserved_source_sha256=hashlib.sha256(oldfile.read_bytes()).hexdigest(),
 checks=checks,all_passed=all(x['passed'] for x in checks),
 original_strict_entropy_error=olderr,repaired_strict_entropy_error=newerr,
 initial_failure_json_preserved=str(Path(__file__).with_name('nc_initial_probe_failure.json')))
Path(__file__).with_name('nc_repair_probe_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
