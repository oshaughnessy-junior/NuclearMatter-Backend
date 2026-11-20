from pathlib import Path
import sys,json,hashlib
P=Path(__file__).resolve().parent;B=P/'package_copy';group=sys.argv[1];sys.path.insert(0,str(next((P/'wheels').glob('*.whl'))));sys.dont_write_bytecode=True
import nmbackend,jax,pytest
jax.config.update('jax_enable_x64',True)
assert '.whl/'in nmbackend.__file__
paths=['test_weak_reference_current.py','test_units.py','test_lepton_derivatives.py']if group=='current_cold'else['test_weak_reference_moments.py']
pins=json.loads((P/'BASE_PIN.json').read_text())['files']
for f in paths:
 p='tests/'+f;assert hashlib.sha256((B/p).read_bytes()).hexdigest()==pins[p]['sha256']
fixture='tests/data/free_weak_reference.json';assert hashlib.sha256((B/fixture).read_bytes()).hexdigest()==pins[fixture]['sha256']
status=pytest.main([*[str(B/'tests'/f)for f in paths],'-q','--junitxml='+str(P/(group+'.xml')),'-p','no:cacheprovider'])
from nmbackend.reference import weak_reference as W
from nmbackend.eos import leptons
assert '.whl/'in W.__file__ and '.whl/'in leptons.__file__
(P/(group+'_origin.json')).write_text(json.dumps(dict(root=nmbackend.__file__,weak=W.__file__,leptons=leptons.__file__,status=status,unchanged_test_paths=paths),indent=2)+'\n');sys.exit(status)
