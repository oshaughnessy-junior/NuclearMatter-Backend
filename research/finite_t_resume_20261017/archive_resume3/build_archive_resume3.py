"""Metadata-only final archive3; no physical package import/call or remote action."""
from pathlib import Path
import json,hashlib,gzip,base64,shutil,subprocess,sys
B=Path('/Users/rossma/Genesis_LANL/outputs/nmb-finite-t-resume-20261017');A=B/'archive_resume3';S=A/'staged';PREFIX='research/finite_t_resume_20261017/';oldroot=B.parent/'nmb-finite-t-20261015'
assert not S.exists();S.mkdir()
sha=lambda raw:hashlib.sha256(raw).hexdigest()
git=lambda raw:hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
roots=['beta_support','beta_review_preparation','review_beta_support'];expected=[('beta_support/MANIFEST.json','acec26c884203ed396e7d9ff0684a31568f172826a692b50a52fb589380547f6'),('beta_support/conditional_beta.py','148c99c37dc0550ca983b53d86123e25125469b15fd2fde9edc16c169c2ac191'),('beta_review_preparation/MANIFEST.json','620ff40628d052f728221f56661b424abaed4ab349b3d1f8569ce5cbacecf62d'),('review_beta_support/MANIFEST.json','61705a2cfd3f38c22facb92a11809f8bc9abc9f759b0975d7c90f65bf1336b4f'),('review_beta_support/REVIEW.md','cedb91c9465bc1cece10bcdcf5f64d5cb28c7287996ff1923f3da8e4e6b5cd50'),('review_beta_support/CHECKPOINT.json','2cae22ff1c83537f94e272364e17d807723ed2d1855c47c53c8bd5eeb9859716')]
for rel,h in expected:assert sha((B/rel).read_bytes())==h,rel
filecounts={}
for root in roots:
 mf=json.loads((B/root/'MANIFEST.json').read_text())['files_sha256'];filecounts[root]=len(mf)
 for rel,h in mf.items():assert sha((B/root/rel).read_bytes())==h,root+'/'+rel
pins=json.loads((B/'beta_support/ORIGINAL_SOURCE_PIN.json').read_text());preppins=json.loads((B/'beta_review_preparation/LOCK.json').read_text())['source_sha256']
for rel,h in dict(pins,**preppins).items():assert sha(Path(rel).read_bytes())==h,rel
initial=json.loads((A/'PREPARATION.json').read_text())
for rel,h in initial['originals'].items():assert sha((B/rel).read_bytes())==h
index={}
for i in range(1,49):
 root=oldroot/f'archive{i}/staged';p=root/f'ARCHIVE{i}_EVIDENCE_MANIFEST.json'
 if p.exists():
  for rel,r in json.loads(p.read_text()).items():index.setdefault(r['original_sha256'],(i,root,rel,r))
def resolve(h):
 assert h in index,'No exact prior archive resolution: '+h
 i,root,rel,r=index[h];packed=(root/r['stored_path']).read_bytes();assert sha(packed)==r['stored_sha256'];raw=gzip.decompress(packed)if r['encoding']=='gzip'else packed;assert sha(raw)==h
 return dict(archive=i,original_path=rel,stored_path=r['stored_path'],sha256=h,research_base='e901158e01d55b2c007a0750bf628006b22d7488')
(A/'SOURCE_RESOLUTION_LEDGER.json').write_text(json.dumps(dict(author={p:resolve(h)for p,h in pins.items()},prep={p:resolve(h)for p,h in preppins.items()},model='Original PR55 physics bytes unchanged; no production repair',publication_head_lease='Root refreshes current backend head; archive2 may be published meanwhile'),indent=2)+'\n')
(A/'RESOURCE_OBSERVATION.json').write_text(json.dumps(dict(utc='2026-10-17T19:45:43Z',account='82f4cfd9-9e20-47cf-9997-761328d89f60',limit='codex',window_minutes=10080,reset=1792652325,used_percent=94,winddown=97,hard=99,exact_identity_verified=True,no_credit_consumption=True),indent=2)+'\n')
extras=[A/'PREPARATION.json',A/'SOURCE_RESOLUTION_LEDGER.json',A/'RESOURCE_OBSERVATION.json',A/'build_archive_resume3.py']+sorted((B/'paper_draft_resume3/initial_pending').glob('*'))
def input_inventory():
 paths=[p for root in roots for p in(B/root).rglob('*')if p.is_file()]+[Path(p)for p in dict(pins,**preppins)]+extras
 return{str(p):sha(p.read_bytes())for p in paths}
before=input_inventory();manifest={};exclusions=[];sources={};canonical={};aliases=[];entries=[];binary=[]
paths=[p for root in roots for p in sorted((B/root).rglob('*'))if p.is_file()]+extras
for p in paths:
 rel=str(p.relative_to(B));raw=p.read_bytes();key=sha(raw);reason=None
 if '__pycache__'in p.parts or p.suffix=='.pyc':reason='Generated interpreter cache; exact hash retained'
 elif 'beta_pr55_reference'in p.parts and key in pins.values():reason='Exact old source namespace copy; '+json.dumps(resolve(key),sort_keys=True)
 if reason:exclusions.append(dict(path=rel,sha256=key,reason=reason));continue
 if key in canonical:record=canonical[key].copy();aliases.append(dict(original=rel,canonical_stored=record['stored_path'],original_sha256=key))
 else:
  try:raw.decode('utf-8');istext=True
  except UnicodeDecodeError:istext=False
  packed=len(raw)>40000 or(p.suffix=='.json'and len(raw)>20000)or not istext;data=gzip.compress(raw,mtime=0)if packed else raw;stored=rel+('.gz'if packed else'');dest=S/stored;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data);record=dict(encoding='gzip'if packed else'utf-8',stored_path=stored,original_sha256=key,stored_sha256=sha(data));canonical[key]=record
 manifest[rel]=record;sources[rel]=p
scope='''# Resumed archive3: selected conditional moving-parent beta reference

DIFFERENT74PASS0newFAIL, four integrals at1.2/new1.27MeV, accepts selected alternate fullfree tree-level coefficients in fixedT2bath. Author44checks/eightcalls include both exact old tail witnesses and alternate1.2/1.3 profiles. Original12beta failures and two reproduced tail failures remain; upper incident-panel12to20/isolated axes/wholeenergy/uniformdomain/belowendpoint/API/transport/AD/empirical accuracy OPEN. No production or candidate edit/import/science/build/Git action by archiver.

SAME literal-current algebra and threshold-parent Laguerre family are SHARED; another namespace is not independent current physics. Independence is rest-Dalitz/cone-delta phase reduction, Jacobi/angular sampling, separate proper-time/spin-two/one-helicity-DOS normalization and actual FD assembly. Direct branches precede KMS; no fitted normalization or zero-by-smallness. No rigorous tail theorem.

All frozen author/preparation/reviewer files and initial pending state verify. Old source namespace copies excluded ONLY with exact SHA/priorarchive resolution; initial pending paper and data are recoverable here. Canonical byte dedup retains originalpaths; deterministicgzip/originalstoredSHA. Restorer validates all sources/destinations before ANYwrite, refuses changed/unsafe/corrupt inputs.

Exclusions:
'''+''.join('\n- '+json.dumps(e,sort_keys=True)for e in exclusions)+'\n'
(S/'ARCHIVE_RESUME3_SCOPE.md').write_text(scope);(S/'ARCHIVE_RESUME3_EVIDENCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
restorer='''"""Verify/restore archive_resume3 evidence; refuse changed files and unsafe paths."""
from pathlib import Path
import gzip,hashlib,json,sys
ROOT=Path(__file__).resolve().parent

def safe(relative):
    rel=Path(relative)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('Unsafe archive-relative path: '+relative)
    path=(ROOT/rel).resolve()
    if ROOT not in path.parents:
        raise ValueError('Path outside archive: '+relative)
    return path

def main():
    args=sys.argv[1:]
    if any(x!='--verify-only' for x in args):
        raise ValueError('Only --verify-only is supported')
    manifest=json.loads((ROOT/'ARCHIVE_RESUME3_EVIDENCE_MANIFEST.json').read_text())
    pending=[];compressed=0
    # Preflight the ENTIRE archive before writing any extracted evidence.
    for original,record in manifest.items():
        source=safe(record['stored_path']);dest=safe(original)
        packed=source.read_bytes()
        if hashlib.sha256(packed).hexdigest()!=record['stored_sha256']:
            raise ValueError('Stored checksum mismatch: '+str(source))
        if record['encoding']=='gzip':
            raw=gzip.decompress(packed);compressed+=1
        elif record['encoding']=='utf-8':
            raw=packed;raw.decode('utf-8')
        else:
            raise ValueError('Unknown archive encoding')
        if hashlib.sha256(raw).hexdigest()!=record['original_sha256']:
            raise ValueError('Original checksum mismatch: '+original)
        if dest.exists() and dest.read_bytes()!=raw:
            raise FileExistsError('Refusing to overwrite changed evidence: '+str(dest))
        if not dest.exists():pending.append((dest,raw))
    if '--verify-only' not in args:
        for dest,raw in pending:
            dest.parent.mkdir(parents=True,exist_ok=True)
            # Exclusive create protects a change appearing after preflight.
            with dest.open('xb') as stream:stream.write(raw)
    print(str(len(manifest))+' originals verified; '+str(compressed)+
          ' compressed; '+str(0 if '--verify-only' in args else len(pending))+' restored')

if __name__=='__main__':main()
'''
(S/'restore_archive_resume3_evidence.py').write_text(restorer)
# Validate every original/stored round trip and then actual restoration in a
# disposable owned fixture; no frozen original or stage file is changed.
for rel,rec in manifest.items():
 stored=(S/rec['stored_path']).read_bytes();raw=gzip.decompress(stored)if rec['encoding']=='gzip'else stored
 assert raw==sources[rel].read_bytes()and sha(raw)==rec['original_sha256']
fixture=A/'roundtrip_fixture';shutil.copytree(S,fixture)
def restore(*args):return subprocess.run([sys.executable,'-B',str(fixture/'restore_archive_resume3_evidence.py'),*args],cwd=A,capture_output=True,text=True,timeout=30)
verify=restore('--verify-only');assert verify.returncode==0,verify.stderr
for rel,rec in manifest.items():
 if rec['encoding']=='gzip':assert not(fixture/rel).exists()
actual=restore();assert actual.returncode==0,actual.stderr
for rel,p in sources.items():assert(fixture/rel).read_bytes()==p.read_bytes()
idempotent=restore();assert idempotent.returncode==0,idempotent.stderr
def clean():
 shutil.rmtree(fixture);shutil.copytree(S,fixture)
def snap():return{str(p.relative_to(fixture)):sha(p.read_bytes())for p in fixture.rglob('*')if p.is_file()}
clean();mm=json.loads((fixture/'ARCHIVE_RESUME3_EVIDENCE_MANIFEST.json').read_text());witness='changed_output_alias_witness.txt';mm[witness]=next(iter(mm.values())).copy();(fixture/'ARCHIVE_RESUME3_EVIDENCE_MANIFEST.json').write_text(json.dumps(mm));changed=fixture/witness;changed.write_bytes(b'changed evidence must survive\n');old=snap();denied=restore();assert denied.returncode!=0 and'Refusing to overwrite changed evidence'in denied.stderr and snap()==old
clean();rec=next(iter(manifest.values()));packed=fixture/rec['stored_path'];packed.write_bytes(packed.read_bytes()+b'poison');old=snap();corrupt=restore();assert corrupt.returncode!=0 and'Stored checksum mismatch'in corrupt.stderr and snap()==old
unsafe_results=[]
for original,stored in [('../archive_resume3_escape_witness.txt',None),('/tmp/archive_resume3_escape_witness.txt',None),('valid_unsafe_source_witness','../source_escape.txt')]:
 clean();mm=json.loads((fixture/'ARCHIVE_RESUME3_EVIDENCE_MANIFEST.json').read_text());rr=next(iter(mm.values())).copy()
 if stored:rr['stored_path']=stored
 mm[original]=rr;(fixture/'ARCHIVE_RESUME3_EVIDENCE_MANIFEST.json').write_text(json.dumps(mm));old=snap();bad=restore();assert bad.returncode!=0 and'Unsafe archive-relative path'in bad.stderr and snap()==old
 unsafe_results.append(dict(original=original,stored_path=stored,stderr=bad.stderr,no_partial_writes=True))
shutil.rmtree(fixture)
# All staged bytes become exact Git blob payloads, including helpers/manifest.
localhash={}
for p in sorted(S.rglob('*')):
 if not p.is_file():continue
 raw=p.read_bytes();path=PREFIX+str(p.relative_to(S));item=dict(path=path,mode='100644',type='blob');localhash[path]=git(raw)
 try:text=raw.decode('utf-8')
 except UnicodeDecodeError:item['base64']=base64.b64encode(raw).decode();binary.append(item)
 else:item['content']=text;entries.append(item)
(A/'entries.json').write_text(json.dumps(entries));(A/'binary.json').write_text(json.dumps(binary));(A/'hashes.json').write_text(json.dumps([dict(path=p,sha=h)for p,h in localhash.items()]));(A/'LOCAL_GIT_HASHES.json').write_text(json.dumps(localhash,indent=2)+'\n')
# Plain-text create-tree payload batches; gzip blobs remain in binary.json.
batch=[];size=2;batches=[]
for item in entries:
 n=len(json.dumps(item).encode())+2
 if batch and size+n>100000:batches.append(batch);batch=[];size=2
 batch.append(item);size+=n
if batch:batches.append(batch)
for i,batch in enumerate(batches):(A/f'payload_{i}.json').write_text(json.dumps(batch))
for item in entries:
 raw=item['content'].encode();assert git(raw)==localhash[item['path']]
for item in binary:
 raw=base64.b64decode(item['base64']);assert git(raw)==localhash[item['path']]
after=input_inventory();assert before==after
assert sha((oldroot/'hot_beta_moments/results.json').read_bytes())=='57e2e89e820b0e80b9d0cd4956d67c54e208cd78d5e2609edcd8e1ef54f94d88'
result=dict(status='FROZEN_SELECTED_BETA_ALTERNATE_NO_UPLOAD',filecounts=filecounts,author_old_source_pins=len(pins),prep_old_source_pins=len(preppins),originals=len(manifest),canonical_payloads=len(canonical),aliases=aliases,exclusions=exclusions,staged_leaves=len(localhash),utf8_entries=len(entries),binary_entries=len(binary),payload_batches=len(batches),roundtrip_originals_verified=len(manifest),verify_stdout=verify.stdout,restore_stdout=actual.stdout,idempotent_stdout=idempotent.stdout,changed_file_refusal=denied.stderr,corrupt_stored_refusal=corrupt.stderr,unsafe_tests=unsafe_results,all_preflight_no_partial_writes=True,all_original_bytes_unchanged=True,input_hashes=before,manifest_sha256=sha((S/'ARCHIVE_RESUME3_EVIDENCE_MANIFEST.json').read_bytes()),restorer_sha256=sha((S/'restore_archive_resume3_evidence.py').read_bytes()),physical_calls=0)
(A/'STAGING_REPORT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items()if k not in ['aliases','exclusions','input_hashes','unsafe_tests','changed_file_refusal','corrupt_stored_refusal']},indent=2))
