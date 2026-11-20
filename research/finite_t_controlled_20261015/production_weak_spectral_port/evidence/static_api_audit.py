from pathlib import Path
import ast,json,sys,hashlib,platform
import numpy,scipy
R=Path(__file__).resolve().parents[1];rows=[]
for p in sorted((R/'staged').rglob('*.py')):
 tree=ast.parse(p.read_text(),feature_version=(3,9));calls=[]
 for n in ast.walk(tree):
  if isinstance(n,(ast.Import,ast.ImportFrom)):
   calls.append(ast.unparse(n))
 rows.append(dict(path=str(p.relative_to(R/'staged')),python39_syntax=True,imports=calls))
sys.path.insert(0,str(R/'test_mirror/src'))
from nmbackend.reference import weak_reference as W
from nmbackend.reference import _weak_current,_weak_quadrature,_weak_spectral_common,_weak_spectral_capture,_weak_spectral_decay
origins={x.__name__:str(Path(x.__file__).resolve())for x in [W,_weak_current,_weak_quadrature,_weak_spectral_common,_weak_spectral_capture,_weak_spectral_decay]}
assert all(str((R/'test_mirror/src').resolve())in p for p in origins.values())
result=dict(python=platform.python_version(),numpy=numpy.__version__,scipy=scipy.__version__,syntax=rows,origins=origins,minimum_API_static='dataclasses3.7; leggauss1.7+; logaddexp long-standingNumPy; scipy.special.logsumexp predates1.7. No scipy.log_expit/new dependency.',minimum_runtime_executed=False,full_CI_executed=False,numpy_validation_pin='current2.3.5 outside separatecampaign<2.3')
(R/'evidence/STATIC_API_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print('syntax',len(rows),'origins',len(origins),'pass')
