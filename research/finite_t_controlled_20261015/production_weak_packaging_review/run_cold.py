from pathlib import Path
import sys,shutil,hashlib,json
P=Path(__file__).resolve().parent;A=P.parent/'production_weak_port';T=P/'cold_tests';T.mkdir(exist_ok=True);provenance={}
for f in ['test_units.py','test_lepton_derivatives.py']:
 p=A/'test_mirror/tests'/f;prior=P.parent/'production_rmf_port/test_mirror/tests'/f;assert p.read_bytes()==prior.read_bytes();shutil.copy2(p,T/f);provenance[f]=hashlib.sha256(p.read_bytes()).hexdigest()
(P/'COLD_SOURCE_PIN.json').write_text(json.dumps(provenance,indent=2)+'\n')
sys.path.insert(0,str(next((P/'wheels').glob('*.whl'))))
import jax;jax.config.update('jax_enable_x64',True)
import nmbackend,pytest
assert '.whl/'in nmbackend.__file__
status=pytest.main([str(T),'-q','--junitxml='+str(P/'cold_tests.xml'),'-p','no:cacheprovider'])
from nmbackend.eos import leptons
assert '.whl/'in leptons.__file__
(P/'cold_origin.json').write_text(json.dumps(dict(root=nmbackend.__file__,leptons=leptons.__file__,status=status),indent=2)+'\n');sys.exit(status)
