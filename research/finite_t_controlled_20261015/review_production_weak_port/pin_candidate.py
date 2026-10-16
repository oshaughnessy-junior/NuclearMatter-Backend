from pathlib import Path
import json,hashlib,shutil,datetime
P=Path(__file__).resolve().parent;A=P.parent/'production_weak_port';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
expected={'CANDIDATE_MANIFEST.json':'795975d51e75e47360822e70845b1eac5ba68df2f134877b7ed1e34ca339789d','MANIFEST.json':'12404beabd3aaf98db37cb2c880ea061cc5d55edd1e813bf8ab8a9c162f11647','AUTHOR_REPORT.md':'7e9b91ee2c593ca5f78ff652a4e6cb380b997b5762e11b44bce217e6056d1a82'}
for n,h in expected.items():assert sha(A/n)==h,n
m=json.loads((A/'CANDIDATE_MANIFEST.json').read_text());assert len(m['files'])==11
full=json.loads((A/'MANIFEST.json').read_text())['sha256'];checks={}
for n,h in full.items():assert sha(A/n)==h,n
for n,h in m['files'].items():
 for base in['staged','test_mirror']:assert sha(A/base/n)==h,(base,n)
checks['candidate_files']=m['files'];checks['author_files']=full
for n,h in json.loads((P/'PREPARATION_MANIFEST.json').read_text())['files'].items():assert sha(P/n)==h,n
for n in expected:shutil.copyfile(A/n,P/('PINNED_'+n))
shutil.copytree(A/'test_mirror',P/'candidate_copy',ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
checks['frozen_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();checks['candidate_source']='owncopy realpackage frozen source, no monkeypatch/stub';(P/'CANDIDATE_PIN.json').write_text(json.dumps(checks,indent=2)+'\n');print('PINNED',len(full),'author paths and11staged/mirrorfiles; purepreparationunchanged')
