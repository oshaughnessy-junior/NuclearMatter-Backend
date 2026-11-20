"""Fresh process: import actual wheel directly, no package/site monkeypatching."""
from pathlib import Path
import sys,json,hashlib,zipfile,math,contextlib,io,runpy,importlib.metadata
import numpy as np
P=Path(__file__).resolve().parent;sys.dont_write_bytecode=True
wheel=next((P/'wheels').glob('*.whl'));checks=[]
def ck(n,v):checks.append(dict(name=n,passed=bool(v)))
with zipfile.ZipFile(wheel)as z:
 content=z.read('nmbackend/reference/finite_t_leptons.py');ck('wheelmoduleexactacceptedbytes',hashlib.sha256(content).hexdigest()=='baeef78b174ecc0bcc1ab6016c839a8e8dc6a2ba915541d7fd994d994f01e719')
 ck('wheelrealpackageinitializers',all(x in z.namelist()for x in['nmbackend/__init__.py','nmbackend/reference/__init__.py','nmbackend/eos/__init__.py']))
 metadata=z.read('nmbackend-0.0.1.dist-info/METADATA').decode();(P/'wheel_METADATA.txt').write_text(metadata)
 ck('unchangednumpyminimum','Requires-Dist: numpy>=1.21'in metadata);ck('unchangedscipyminimum','Requires-Dist: scipy>=1.7'in metadata)
sys.path.insert(0,str(wheel))
import nmbackend
from nmbackend import constants
from nmbackend.reference import finite_t_leptons as L
from nmbackend.eos.leptons import lepton_sector
import jax
jax.config.update('jax_enable_x64',True)
ck('toppackagefromexactwheel',str(nmbackend.__file__).startswith(str(wheel)));ck('referencefromexactwheel',str(L.__file__).startswith(str(wheel)));ck('constantsfromexactwheel',str(constants.__file__).startswith(str(wheel)))
ck('publicmodulefourdeclaredfunctions',L.__all__==['LeptonState','state_mu','state_density','thermal_energy_density']);ck('immutableLeptonState',L.LeptonState.__dataclass_params__.frozen)
s=L.state_density(1.7,3e-7);ck('wheelfreshsignednetroot',abs(s.net_density/3e-7-1)<1e-9);ck('wheelCPUfloatscalarstate',all(isinstance(v,(float,np.floating))and np.ndim(v)==0 for v in s.asdict().values()));ck('wheelfinitethermal',L.thermal_energy_density(s)>0)
try:L.state_mu(0.,1.);reject=True
except ValueError:reject=True
else:reject=False
ck('T0doesnotreplacecoldAPI',reject)
cold=lepton_sector(.01,muons=False);ck('realcoldpackagefunctionpositive',float(cold[0])>0 and float(cold[3])>0);ck('realcoldchargeclosure',abs(float(cold[1])/.01-1)<1e-12 and float(cold[2])==0)
buf=io.StringIO()
with contextlib.redirect_stdout(buf):runpy.run_path(str(P/'package_copy/examples/finite_t_leptons.py'),run_name='__main__')
(P/'example_stdout.json').write_text(buf.getvalue());ex=json.loads(buf.getvalue());ck('actualexamplewheelstates',abs(ex['electron']['net_density']/1e-6-1)<1e-9 and abs(ex['muon']['net_density']/.005-1)<1e-9);ck('exampleindependentflavormu',ex['electron']['chemical_potential']!=ex['muon']['chemical_potential']);ck('exampleexplicitMeVunits',ex['units']['temperature_chemical_potential_mass']=='MeV');ck('exampleexplicitnetnumbersign','charge=-e*n_net'in ex['net_number_sign'])
versions={n:importlib.metadata.version(n)for n in['numpy','scipy','jax','setuptools','wheel']}
out=dict(status='PASS'if all(c['passed']for c in checks)else'FAIL',checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],wheel=str(wheel),wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),module_path=str(L.__file__),cold_function_module=lepton_sector.__module__,versions=versions,installation_performed=False,site_or_package_stubs=False,minimum_runtime_tested=False)
(P/'wheel_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out['status'],out['passed'],out['failed'])
