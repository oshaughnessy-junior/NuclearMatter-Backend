"""Verify exact tracked package blobs using clean local base + GitHub head diff."""
from pathlib import Path
import subprocess,os,json,hashlib
P=Path(__file__).resolve().parents[1]
OLD=Path('/Users/rossma/unixhome/Projects/LIGO-ScienceMode/communications/20260602-Me-UpdatedRIFTMatterBackend/NuclearMatter-Backend')
env=dict(os.environ,GIT_OPTIONAL_LOCKS='0')
def git(*args):return subprocess.run(['git','-c','core.fsmonitor=false',*args],cwd=OLD,env=env,check=True,capture_output=True,text=True).stdout
assert git('rev-parse','HEAD').strip()=='5e0208cccf386c723f10955d0c81390fa1ecb972'
assert git('status','--porcelain','--untracked-files=no')==''
expected={}
for row in git('ls-files','--stage','src').splitlines():
 prefix,path=row.split('\t');expected[path]=prefix.split()[1]
records=json.loads((P/'evidence/github_source_records.json').read_text())
for r in records:
 if r['path'].startswith('src/'):expected[r['path']]=r['sha']
def blob(q):
 b=q.read_bytes();return hashlib.sha1(('blob '+str(len(b))+'\0').encode()+b).hexdigest()
checks=[]
for rel,h in expected.items():checks.append(dict(path=rel,expected_git_blob=h,actual_git_blob=blob(P/'test_mirror'/rel),passed=blob(P/'test_mirror'/rel)==h))
assert all(c['passed']for c in checks)
# Exact cold-test and candidate-overlay identities, separately from unchanged package.
for r in records:
 if r['path'].startswith('tests/')or r['path']=='pyproject.toml':assert blob(P/'test_mirror'/r['path'])==r['sha']
new='src/nmbackend/reference/finite_t_leptons.py';assert (P/'test_mirror'/new).read_bytes()==(P/'staged'/new).read_bytes()
(P/'evidence/mirror_verification.json').write_text(json.dumps(dict(local_clean_base='5e0208cccf386c723f10955d0c81390fa1ecb972',github_target='ffb5cc72949f2e5401251847aef4c9c7a98dde78',checks=checks,passed=len(checks),candidate_overlay=new,actual_package_initializers_preserved=True),indent=2)+'\n')
print('Exact head package files:',len(checks),'all PASS; real initializers preserved')
