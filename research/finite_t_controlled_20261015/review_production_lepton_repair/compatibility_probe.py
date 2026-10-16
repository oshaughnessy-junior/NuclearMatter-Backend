"""Missing-export simulation supplements primary static SciPy1.7 mismatch evidence.

This is not execution of SciPy1.7. The existing SciPy export is restored.
"""
from pathlib import Path
import sys,importlib.util,json,hashlib
P=Path(__file__).resolve().parent;A=P.parent/'production_lepton_port';sys.dont_write_bytecode=True;sys.path.insert(0,str(A/'test_mirror/src'))
import nmbackend.reference
import scipy.special as special
saved=special.log_expit
try:
 del special.log_expit
 sp=importlib.util.spec_from_file_location('nmbackend.reference.missing_minimum_API_probe',A/'staged/src/nmbackend/reference/finite_t_leptons.py');m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m
 try:sp.loader.exec_module(m);error=None
 except ImportError as exc:error=str(exc)
finally:special.log_expit=saved
checks=[]
for row in json.loads((A/'evidence/mirror_verification.json').read_text())['checks']:
 q=A/'test_mirror'/row['path'];b=q.read_bytes();blob=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest();checks.append(dict(name='baseline Git blob '+row['path'],passed=blob==row['expected_git_blob']))
compat=dict(name='declared SciPy>=1.7 API compatibility',passed=False,actual_import_error_when_export_missing=error,primary_evidence='https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.log_expit.html states Added in version1.8.0',scope='Static required-API incompatibility plus a controlled missing-export simulation; not execution underSciPy1.7')
assert error and 'log_expit'in error
out=dict(checks=checks,passed=sum(x['passed']for x in checks),compatibility_failures=[compat],export_restored=special.log_expit is saved);(P/'compatibility_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out['passed'],compat)
