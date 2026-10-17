from pathlib import Path
import json,hashlib,shutil,subprocess,os
P=Path(__file__).resolve().parent;A=P.parent/'production_lepton_port';B=P/'package_copy';OLD=Path('/Users/rossma/unixhome/Projects/LIGO-ScienceMode/communications/20260602-Me-UpdatedRIFTMatterBackend/NuclearMatter-Backend')
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
manifest=json.loads((A/'CANDIDATE_MANIFEST.json').read_text());assert sha(A/'CANDIDATE_MANIFEST.json')=='5e9a2793bc0de9c2e2a01c78061bd7a76f093b260cb8a7564971aae18a1bf282'
if B.exists():raise RuntimeError('Immutable build preparation exists')
shutil.copytree(A/'test_mirror/src',B/'src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'));shutil.copy2(A/'test_mirror/pyproject.toml',B/'pyproject.toml')
records=json.loads((A/'evidence/github_source_records.json').read_text());pyrec=next(r for r in records if r['path']=='pyproject.toml');blob=lambda b:hashlib.sha1(('blob '+str(len(b))+'\0').encode()+b).hexdigest();assert blob((B/'pyproject.toml').read_bytes())==pyrec['sha']
checks=[]
for r in json.loads((A/'evidence/mirror_verification.json').read_text())['checks']:
 checks.append(dict(path=r['path'],expected_blob=r['expected_git_blob'],actual_blob=blob((B/r['path']).read_bytes())));assert checks[-1]['actual_blob']==checks[-1]['expected_blob']
compare=json.loads((A/'evidence/local_to_head_compare.json').read_text())['structuredContent'];changed=[f['filename']for f in compare['files']];assert 'LICENSE'not in changed and not compare['too_large']
env=dict(os.environ,GIT_OPTIONAL_LOCKS='0');license_bytes=subprocess.run(['git','-c','core.fsmonitor=false','show','5e0208cccf386c723f10955d0c81390fa1ecb972:LICENSE'],cwd=OLD,env=env,capture_output=True,check=True).stdout;(B/'LICENSE').write_bytes(license_bytes)
for path,h in manifest['files'].items():
 assert sha(A/'staged'/path)==h;f=B/path;f.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(A/'staged'/path,f)
(P/'PREPARATION.json').write_text(json.dumps(dict(base_commit='ffb5cc72949f2e5401251847aef4c9c7a98dde78',source_package_blob_checks=checks,package_source_count=len(checks),candidate_paths=manifest['files'],pyproject_git_blob=pyrec['sha'],license_sha256=hashlib.sha256(license_bytes).hexdigest(),license_provenance='Exact git show old clean pinned5e0208 blob; complete compare toheadhasnoLICENSEchange',compare_file_count=len(changed),source_sha256=sha(B/'src/nmbackend/reference/finite_t_leptons.py'),build='Existing setuptools.build_meta.build_wheel directly, no isolation/install/network dependencies'),indent=2)+'\n');print(len(checks),'source blobs +5 exactcandidatepaths prepared')
