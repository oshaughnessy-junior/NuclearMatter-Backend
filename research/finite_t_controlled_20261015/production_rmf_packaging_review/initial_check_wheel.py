from pathlib import Path
import sys,json,hashlib,zipfile,math,io,contextlib,runpy,ast,importlib.metadata,time
import numpy as np
P=Path(__file__).resolve().parent;A=P.parent/'production_rmf_port';wheel=next((P/'wheels').glob('*.whl'));checks=[];start=time.monotonic();sys.dont_write_bytecode=True
sha=lambda b:hashlib.sha256(b).hexdigest()
def ck(name,value):checks.append(dict(name=name,passed=bool(value)))
m=json.loads((A/'CANDIDATE_MANIFEST.json').read_text());base=json.loads((P/'PREPARATION.json').read_text());omissions=[]
with zipfile.ZipFile(wheel)as z:
 names=z.namelist()
 for path,h in m['files'].items():
  raw=(P/'package_copy'/path).read_bytes();ck('overlay exact '+path,sha(raw)==h)
  if path.endswith('.py'):ast.parse(raw.decode(),feature_version=(3,9));ck('Python3.9 grammar '+path,True)
  if path.startswith('src/'):
   ck('wheel exact module '+path,sha(z.read(path[4:]))==h)
 for r in base['baseline_blobs']:
  if r['path'][4:]not in names:omissions.append(r['path']);ck('baseline packaging omission nonPython '+r['path'],not r['path'].endswith('.py'));continue
  b=z.read(r['path'][4:]);ck('baseline wheel unchanged '+r['path'],hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==r['blob'])
 ck('no unmerged lepton module',not any('finite_t_leptons' in n for n in names));ck('no unmerged baryon module',not any('finite_t_baryons'in n for n in names))
 ck('only three new package Python modules',sorted(n for n in names if n.startswith('nmbackend/')and n.endswith('.py')and 'src/'+n not in [r['path']for r in base['baseline_blobs']])==['nmbackend/reference/_rmf_moments.py','nmbackend/reference/_rmf_response.py','nmbackend/reference/finite_t_rmf.py'])
 md=z.read('nmbackend-0.0.1.dist-info/METADATA').decode();(P/'wheel_METADATA.txt').write_text(md)
 for line in ['Requires-Python: >=3.9','Requires-Dist: numpy>=1.21','Requires-Dist: scipy>=1.7']:ck('metadata '+line,line in md)
 ck('README new reference discoverable','finite_t_rmf'in md)
 ck('docs example fixture repository-only under baseline wheel policy',all(n not in names for n in ['docs/finite_temperature_rmf.md','examples/finite_t_rmf.py','tests/data/finite_t_gm1_reference.json']))
 (P/'wheel_filelist.json').write_text(json.dumps(names,indent=2)+'\n')
sys.path.insert(0,str(wheel))
import nmbackend
from nmbackend import constants as C
from nmbackend.reference import finite_t_rmf as B
from nmbackend.reference import _rmf_moments as M,_rmf_response as R
import jax;jax.config.update('jax_enable_x64',True)
from nmbackend.eos.leptons import lepton_sector
from nmbackend.eos.rmft import RMFTEOS,GM1
for name,module in [('root',nmbackend),('constants',C),('public',B),('moments',M),('response',R)]:ck(name+' wheel import',str(module.__file__).startswith(str(wheel)))
ck('no research runtime imports',not any('outputs/nmb-finite-t' in str(getattr(x,'__file__',''))for x in list(sys.modules.values())if x is not None))
ck('explicit public API',B.__all__==['SolveControls','UniformRMFState','state_density','diagnose_density']);ck('immutable state',B.UniformRMFState.__dataclass_params__.frozen)
f=io.StringIO()
with contextlib.redirect_stdout(f):runpy.run_path(str(P/'package_copy/examples/finite_t_rmf.py'),run_name='__main__')
(P/'example_stdout.json').write_text(f.getvalue());ex=json.loads(f.getvalue())
for index,s in enumerate(ex['states']):
 ck('example admitted '+str(index),s['locally_admitted']and s['hessian_min_eigenvalue']>0)
 ck('example energy closure '+str(index),abs(s['energy_density']-s['kinetic_energy_density']-s['scalar_field_energy_density']-s['vector_field_energy_density'])<2e-11)
 ck('example free energy '+str(index),abs(s['free_energy_density']-s['energy_density']+s['temperature']*s['entropy_density'])<2e-11)
 ck('example independent composition '+str(index),abs(s['proton_density']/s['baryon_density']-s['proton_fraction'])<2e-10)
 ck('example actual roots '+str(index),s['normalized_equation_residual']<2e-9)
spin=ex['diagnostic_only_unadmitted'];ck('spinodal diagnostic unadmitted',not spin['locally_admitted']and spin['hessian_min_eigenvalue']<0)
try:B.state_density(1.,.06,.1)
except ValueError:ck('strict spinodal rejection',True)
else:ck('strict spinodal rejection',False)
for args in [(True,.16,.1),(10.,.16,.0),(0.,.16,.1)]:
 try:B.state_density(*args)
 except ValueError:ck('invalid contract '+str(args),True)
 else:ck('invalid contract '+str(args),False)
c=lepton_sector(.0123,muons=False);ck('existing cold lepton API',abs(float(c[1])/.0123-1)<1e-12 and float(c[2])==0 and float(c[3])>0)
ck('cold RMFT class and parameters',callable(RMFTEOS)and GM1==dict(M_mev=939.,C_sigma=11.785,C_omega=7.148,C_rho=4.410,b=.002947,c=-.001070))
ck('no eager root reference export',not hasattr(nmbackend,'state_density'))
pins=json.loads((P/'BEFORE_PIN.json').read_text());changed=[]
for kind,root in [('author',A),('review',P.parent/'review_production_rmf_port')]:
 changed +=[kind+'/'+p for p,h in pins[kind].items()if sha((root/p).read_bytes())!=h]
ck('all frozen author and review unchanged',not changed)
out=dict(status='PASS'if all(c['passed']for c in checks)else'FAIL',passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],checks=checks,baseline_packaging_omissions=omissions,changed=changed,wheel_sha256=sha(wheel.read_bytes()),wheel=str(wheel),module_origin=B.__file__,python=sys.version,versions={x:importlib.metadata.version(x)for x in ['numpy','scipy','jax','diffrax','setuptools','wheel']},elapsed_seconds=time.monotonic()-start,minimum_runtime_executed=False,installations=False,network=False)
(P/'wheel_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out['status'],out['passed'],out['failed'],out['elapsed_seconds']);sys.exit(0 if out['status']=='PASS'else 1)
