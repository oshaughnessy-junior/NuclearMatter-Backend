from pathlib import Path
import json,hashlib,shutil
P=Path(__file__).resolve().parent;R=P.parent;A=R/'production_baryon_port';B=P/'package_copy';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();blob=lambda b:hashlib.sha1(('blob '+str(len(b))+'\0').encode()+b).hexdigest();lock=json.loads((P/'LOCK.json').read_text())
shutil.copytree(A/'test_mirror/src',B/'src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
checks=[]
for row in json.loads((A/'evidence/baseline_package_blobs.json').read_text())['checks']:
 actual=blob((B/row['path']).read_bytes());assert actual==row['expected_git_blob'];checks.append(dict(path=row['path'],blob=actual))
for species,source in [('lepton',R/'production_lepton_port/staged/src/nmbackend/reference/finite_t_leptons.py'),('baryon',A/'staged/src/nmbackend/reference/finite_t_baryons.py')]:
 assert sha(source)==lock['modules'][species];shutil.copy2(source,B/'src/nmbackend/reference'/source.name)
(P/'PACKAGE_PIN.json').write_text(json.dumps(dict(base='ffb5cc72949f2e5401251847aef4c9c7a98dde78',baseline71=checks,overlay=lock['modules'],all_python={str(p.relative_to(B)):sha(p)for p in B.rglob('*.py')},lock_sha256=sha(P/'LOCK.json'),environment='actual owned source package, no stubs/installation/dependencies'),indent=2)+'\n');print('71 baseline blobs +2 exact reviewed module overlays')
