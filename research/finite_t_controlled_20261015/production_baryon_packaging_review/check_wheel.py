from pathlib import Path
import sys,json,hashlib,zipfile,math,io,contextlib,runpy,ast,importlib.metadata
import numpy as np
P=Path(__file__).resolve().parent; A=P.parent/'production_baryon_port';sys.dont_write_bytecode=True;wheel=next((P/'wheels').glob('*.whl'));checks=[]
def ck(name,passed):checks.append(dict(name=name,passed=bool(passed)))
manifest=json.loads((A/'CANDIDATE_MANIFEST.json').read_text()); base=json.loads((P/'PREPARATION.json').read_text())
with zipfile.ZipFile(wheel)as z:
 content=z.read('nmbackend/reference/finite_t_baryons.py');ck('module exact candidate SHA',hashlib.sha256(content).hexdigest()==manifest['files']['src/nmbackend/reference/finite_t_baryons.py']); ast.parse(content.decode(),feature_version=(3,9));ck('Python3.9 grammar',True)
 for r in base['baseline_blobs']:
  if r['path'][4:] not in z.namelist():
   ck('preexisting packaging omission '+r['path'],not r['path'].endswith('.py'));continue
  b=z.read(r['path'][4:]);actual=hashlib.sha1(('blob '+str(len(b))+'\0').encode()+b).hexdigest();ck('unchanged baseline wheel '+r['path'],actual==r['blob'])
 md=z.read('nmbackend-0.0.1.dist-info/METADATA').decode();(P/'wheel_METADATA.txt').write_text(md)
 for line in ['Requires-Python: >=3.9','Requires-Dist: numpy>=1.21','Requires-Dist: scipy>=1.7']:ck('metadata '+line,line in md)
 ck('README candidate included', 'Particle-only finite-temperature baryon' in md or 'finite_t_baryons' in md)
 ck('reference initializer packaged','nmbackend/reference/__init__.py'in z.namelist())
sys.path.insert(0,str(wheel))
import nmbackend
from nmbackend import constants as C
from nmbackend.reference import finite_t_baryons as B
from nmbackend.eos.leptons import lepton_sector
import jax;jax.config.update('jax_enable_x64',True)
for name,m in [('nmbackend',nmbackend),('constants',C),('baryon',B)]:ck(name+' actual wheel import',str(m.__file__).startswith(str(wheel)))
ck('public API narrow',B.__all__==['ParticleState','state_mu','state_density']);ck('immutable state',B.ParticleState.__dataclass_params__.frozen);ck('NMB neutron mass exact',C.M_N==939.565421);ck('NMB proton mass exact',C.M_P==938.272088)
for mass,n in [(C.M_N,7e-8),(C.M_P,3e-8),(939.56542052,.002),(938.27208816,.001)]:
 s=B.state_density(2.3,n,mass_mev=mass);ck('actualwheel fresh root '+str((mass,n)),abs(s.particle_density/n-1)<1e-9);ck('rest accounting '+str((mass,n)),abs((s.energy_density-mass*s.particle_density)/s.kinetic_energy_density-1)<1e-11);ck('free energy '+str((mass,n)),abs((s.energy_density-s.temperature*s.entropy_density-s.free_energy_density)/(abs(s.energy_density)+s.temperature*s.entropy_density))<2e-13);ck('scalar finite fields '+str((mass,n)),all(isinstance(v,(float,np.floating))and np.ndim(v)==0 and math.isfinite(v)for v in s.asdict().values()))
for args in [(0.,1e-7),(2.,0.),(2.,-1e-7),(True,1e-7),(2.,float('nan'))]:
 try:B.state_density(*args)
 except ValueError:ck('invalid state reject '+str(args),True)
 else:ck('invalid state reject '+str(args),False)
c=lepton_sector(.0123,muons=False);ck('cold JAX API charge closure',abs(float(c[1])/.0123-1)<1e-12 and float(c[2])==0 and float(c[3])>0)
f=io.StringIO()
with contextlib.redirect_stdout(f):runpy.run_path(str(P/'package_copy/examples/finite_t_baryons.py'),run_name='__main__')
(P/'example_stdout.json').write_text(f.getvalue());ex=json.loads(f.getvalue());ck('example wheel neutron',abs(ex['neutron']['particle_density']/.001-1)<1e-9);ck('example wheel proton',abs(ex['proton']['particle_density']/.0002-1)<1e-9);ck('example explicit units','MeV fm^-3'in ex['units']);ck('example species independent',ex['neutron']['chemical_potential']!=ex['proton']['chemical_potential'])
pins=json.loads((P/'BEFORE_PIN.json').read_text());changed=[p for p,h in pins.items() if hashlib.sha256((A/p).read_bytes()).hexdigest()!=h];ck('all pinned author files unchanged',not changed)
out=dict(status='PASS'if all(c['passed']for c in checks)else'FAIL',passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],checks=checks,author_changed=changed,wheel=str(wheel),wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),module_origin=B.__file__,versions={x:importlib.metadata.version(x)for x in['numpy','scipy','jax','diffrax','setuptools','wheel']},python=sys.version,installation_performed=False,network_used=False,min_runtime_executed=False)
(P/'wheel_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out['status'],out['passed'],out['failed'])
