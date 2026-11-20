"""Lossless binder evidence archive; metadata-only, no scientific imports/calls."""
from pathlib import Path
import json,hashlib,gzip,base64,shutil,subprocess,sys
B=Path('/Users/rossma/Genesis_LANL/outputs/nmb-finite-t-resume-20261017');A=B/'archive_resume2';S=A/'staged';PREFIX='research/finite_t_resume_20261017/'
assert not S.exists();S.mkdir()
sha=lambda raw:hashlib.sha256(raw).hexdigest()
git=lambda raw:hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
roots=['free_source_binder','review_free_source_binder'];author=B/roots[0];review=B/roots[1];oldroot=B.parent/'nmb-finite-t-20261015'
am=json.loads((author/'MANIFEST.json').read_text());rm=json.loads((review/'MANIFEST.json').read_text())
assert sha((author/'MANIFEST.json').read_bytes())=='7b40760f63ce34a9633a4f6d95d65724e51d82a7ab57b6649077ea3a47586fbf'
assert sha((review/'MANIFEST.json').read_bytes())=='bca67eaeb7566fd621eeca58cff6568ae811c8cee913b15777fb7dcf067f8efa'
assert sha((review/'REVIEW.md').read_bytes())=='9200e61e6970223096ac8386977e6bc960ab486ed0c4a79cd4682e01cd23f0f0'
assert sha((review/'CHECKPOINT.json').read_bytes())=='81f7e9b557a3a0ea036dfcc5e9942fbd27c0533ac4884a5915f12bd9ed2588a8'
for rel,h in am.items():assert sha((author/rel).read_bytes())==h,rel
for rel,h in rm['files'].items():assert sha((review/rel).read_bytes())==h,rel
repo=B.parents[1]
for rel,h in rm['source_pins'].items():assert sha((repo/rel).read_bytes())==h,rel
base=json.loads((author/'STAGED_BASE_PIN.json').read_text());changed=json.loads((author/'CHANGED_PATHS.json').read_text());prereq=json.loads((author/'PREREQUISITES.json').read_text())
assert len(changed['changed'])==8 and changed['unchanged_baseline_files']==85
for rel,h in base.items():
 if rel not in changed['changed']:assert sha((author/'staged'/rel).read_bytes())==h
for rel,h in changed['hashes'].items():assert sha((author/'staged'/rel).read_bytes())==h
for module,rec in prereq['explicit_unmerged_dependency_union'].items():assert sha((oldroot/rec['source']).read_bytes())==rec['sha256']
index={}
for i in range(1,49):
 root=oldroot/f'archive{i}/staged';p=root/f'ARCHIVE{i}_EVIDENCE_MANIFEST.json'
 if p.exists():
  for rel,r in json.loads(p.read_text()).items():index.setdefault(r['original_sha256'],(i,root,rel,r))
basepr='c17472b8a84655380ab64973ef883a6cd5fb2346'
def resolve(h,rel=None):
 if h in index:
  i,root,path,r=index[h];packed=(root/r['stored_path']).read_bytes();assert sha(packed)==r['stored_sha256'];raw=gzip.decompress(packed)if r['encoding']=='gzip'else packed;assert sha(raw)==h
  return dict(archive=i,original_path=path,stored_path=r['stored_path'],sha256=h,available_in_research_base='e901158e01d55b2c007a0750bf628006b22d7488')
 for module,r in prereq['explicit_unmerged_dependency_union'].items():
  if h==r['sha256']:return dict(source_path=r['source'],sha256=h,pr_heads=prereq['PR_heads'],resolution='Explicit frozen prerequisite union')
 assert rel is not None
 return dict(commit=basepr,source_path=rel,sha256=h,resolution='Unchanged PR54 prerequisite baseline path; exact staged baseline and review pins verified')
ledger=dict(status='Explicit staged prerequisite union NOT main',baseline={rel:resolve(h,rel)for rel,h in base.items()if rel not in changed['changed']},prerequisite_modules=prereq,changed_paths=changed,review_source_pins_verified=len(rm['source_pins']),prior_backend_base='e901158e01d55b2c007a0750bf628006b22d7488',remote_head_lease='Root refreshes before publication',physical_calls=0)
(A/'PREREQUISITE_RESOLUTION.json').write_text(json.dumps(ledger,indent=2)+'\n')
(A/'ARCHIVER_RESOURCE_OBSERVATION.json').write_text(json.dumps(dict(utc='2026-10-17T19:33:33Z',used_percent=93,account='82f4cfd9-9e20-47cf-9997-761328d89f60',limit='codex',window_minutes=10080,reset_unix=1792652325,winddown=97,hard=99,exact_identity_reset_verified=True,credits_consumed=False),indent=2)+'\n')
extras=[A/'PREREQUISITE_RESOLUTION.json',A/'ARCHIVER_RESOURCE_OBSERVATION.json',A/'build_archive_resume2.py',A/'pre_payload_size_adjustment/CHECKPOINT.json',A/'pre_payload_size_adjustment/STAGING_REPORT.json',A/'pre_payload_size_adjustment/build_archive_resume2.py']
def input_inventory():
 paths=[p for root in roots for p in(B/root).rglob('*')if p.is_file()]
 paths +=[repo/r for r in rm['source_pins']]+[oldroot/r['source']for r in prereq['explicit_unmerged_dependency_union'].values()]+extras
 return{str(p):sha(p.read_bytes())for p in paths}
before=input_inventory();manifest={};exclusions=[];sources={};entries=[];binary=[];canonical={};aliases=[]
for root in roots:
 for p in sorted((B/root).rglob('*')):
  if not p.is_file():continue
  rel=str(p.relative_to(B));local=str(p.relative_to(B/root));reason=None;resolution=None
  if '__pycache__'in p.parts or '.pytest_cache'in p.parts or any(x.endswith('.egg-info')for x in p.parts)or 'build'in p.parts or p.suffix=='.pyc':reason='Generated build/interpreter/test metadata; exact hash retained'
  elif any(x in p.parts for x in ['wheel','wheel_unpacked','wheel_import']):reason='Duplicate complete wheel/build copy; exact wheel SHA/member/source-origin proof in archived author/reviewer reports'
  elif root==roots[0]and local.startswith('staged/')and local[7:]in base and local[7:]not in changed['changed']:
   assert sha(p.read_bytes())==base[local[7:]];reason='Exact unchanged baseline prerequisite';resolution=resolve(sha(p.read_bytes()),local[7:])
  if reason:exclusions.append(dict(path=rel,sha256=sha(p.read_bytes()),reason=reason,resolution=resolution));continue
  raw=p.read_bytes();key=sha(raw)
  if key in canonical:record=canonical[key].copy();aliases.append(dict(original=rel,canonical_stored=record['stored_path'],original_sha256=key))
  else:
   try:raw.decode('utf-8');istext=True
   except UnicodeDecodeError:istext=False
   packed=len(raw)>40000 or(p.suffix=='.json'and len(raw)>20000)or not istext;data=gzip.compress(raw,mtime=0)if packed else raw;stored=rel+('.gz'if packed else'');dest=S/stored;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data);record=dict(encoding='gzip'if packed else'utf-8',stored_path=stored,original_sha256=key,stored_sha256=sha(data));canonical[key]=record
  manifest[rel]=record;sources[rel]=p
for p in extras:
 rel=str(p.relative_to(B));raw=p.read_bytes();key=sha(raw);packed=len(raw)>40000 or(p.suffix=='.json'and len(raw)>20000);data=gzip.compress(raw,mtime=0)if packed else raw;stored=rel+('.gz'if packed else'');dest=S/stored;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data);record=dict(encoding='gzip'if packed else'utf-8',stored_path=stored,original_sha256=key,stored_sha256=sha(data));canonical[key]=record;manifest[rel]=record;sources[rel]=p
scope='''# Resumed archive2: selected free-matter/source binder implementation

DIFFERENT review294PASS0candidateFAIL plus13actual-wheel cold tests accepts selected scalar CPU free-npe/pair and included EC+PC+decay/inverse closure. This is an eight-path candidate against an explicit unmerged PR51/52/54 prerequisite union, NOT code on main. It supersedes design-only status solely for this selected implementation; old contract claims remain historical. No new source/model/import/build/science/Git action performed by archiver.

All126 author manifest members, reviewer files/sourcepins and85 unchanged baseline files verify. Preserve all unique old guard-completion versions, original JSON serialization failures, initial review scope-fixture failure and wheel SyntaxError. Generated artifacts and complete wheel copies exclude only exact hash-ledger entries; unique source/evidence remains. Runtime/minimum supported CI/clean dependency resolution, broad physical accuracy, uniform-domain admission, ODE/hydro/AD/wholeweak inventory remain OPEN.

Canonical raw-byte dedup retains every admitted original path/hash, original/stored SHA and deterministic gzip mtime0. Safe restorer validates ALL sources and destination preflight before writing; changed destinations/corruption/unsafe paths reject without partial writes. Resource metadata remains backend-only.

Exclusions and exact resolutions follow:
'''+'\nExact exclusions are retained in ARCHIVE_RESUME2_EXCLUSIONS.json.gz (deterministic gzip UTF-8 JSON).\n'
exclusion_raw=(json.dumps(exclusions,indent=2)+'\n').encode();exclusion_gzip=gzip.compress(exclusion_raw,mtime=0);assert gzip.decompress(exclusion_gzip)==exclusion_raw;(S/'ARCHIVE_RESUME2_EXCLUSIONS.json.gz').write_bytes(exclusion_gzip)
(S/'ARCHIVE_RESUME2_SCOPE.md').write_text(scope);(S/'ARCHIVE_RESUME2_EVIDENCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
restorer='''"""Verify/restore archive_resume2 evidence; refuse changed files and unsafe paths."""
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
    manifest=json.loads((ROOT/'ARCHIVE_RESUME2_EVIDENCE_MANIFEST.json').read_text())
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
(S/'restore_archive_resume2_evidence.py').write_text(restorer)
# Validate every original/stored round trip and then actual restoration in a
# disposable owned fixture; no frozen original or stage file is changed.
for rel,rec in manifest.items():
 stored=(S/rec['stored_path']).read_bytes();raw=gzip.decompress(stored)if rec['encoding']=='gzip'else stored
 assert raw==sources[rel].read_bytes()and sha(raw)==rec['original_sha256']
fixture=A/'roundtrip_fixture';shutil.copytree(S,fixture)
def restore(*args):return subprocess.run([sys.executable,'-B',str(fixture/'restore_archive_resume2_evidence.py'),*args],cwd=A,capture_output=True,text=True,timeout=30)
verify=restore('--verify-only');assert verify.returncode==0,verify.stderr
for rel,rec in manifest.items():
 if rec['encoding']=='gzip':assert not(fixture/rel).exists()
actual=restore();assert actual.returncode==0,actual.stderr
for rel,p in sources.items():assert(fixture/rel).read_bytes()==p.read_bytes()
idempotent=restore();assert idempotent.returncode==0,idempotent.stderr
def clean():
 shutil.rmtree(fixture);shutil.copytree(S,fixture)
def snap():return{str(p.relative_to(fixture)):sha(p.read_bytes())for p in fixture.rglob('*')if p.is_file()}
clean();mm=json.loads((fixture/'ARCHIVE_RESUME2_EVIDENCE_MANIFEST.json').read_text());witness='changed_output_alias_witness.txt';mm[witness]=next(iter(mm.values())).copy();(fixture/'ARCHIVE_RESUME2_EVIDENCE_MANIFEST.json').write_text(json.dumps(mm));changed=fixture/witness;changed.write_bytes(b'changed evidence must survive\n');old=snap();denied=restore();assert denied.returncode!=0 and'Refusing to overwrite changed evidence'in denied.stderr and snap()==old
clean();rec=next(iter(manifest.values()));packed=fixture/rec['stored_path'];packed.write_bytes(packed.read_bytes()+b'poison');old=snap();corrupt=restore();assert corrupt.returncode!=0 and'Stored checksum mismatch'in corrupt.stderr and snap()==old
unsafe_results=[]
for original,stored in [('../archive_resume2_escape_witness.txt',None),('/tmp/archive_resume2_escape_witness.txt',None),('valid_unsafe_source_witness','../source_escape.txt')]:
 clean();mm=json.loads((fixture/'ARCHIVE_RESUME2_EVIDENCE_MANIFEST.json').read_text());rr=next(iter(mm.values())).copy()
 if stored:rr['stored_path']=stored
 mm[original]=rr;(fixture/'ARCHIVE_RESUME2_EVIDENCE_MANIFEST.json').write_text(json.dumps(mm));old=snap();bad=restore();assert bad.returncode!=0 and'Unsafe archive-relative path'in bad.stderr and snap()==old
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
result=dict(status='FROZEN_SELECTED_BINDER_STAGE_NO_UPLOAD',author_manifest_verified=len(am),review_manifest_verified=len(rm['files']),review_source_pins_verified=len(rm['source_pins']),unchanged_baseline_verified=85,candidate_paths=8,archived_originals=len(manifest),canonical_payloads=len(canonical),aliases=aliases,staged_leaves=len(localhash),utf8_entries=len(entries),binary_entries=len(binary),payload_batches=len(batches),exclusions=exclusions,all_source_bytes_unchanged=True,roundtrip_originals_verified=len(manifest),verify_stdout=verify.stdout,restore_stdout=actual.stdout,idempotent_stdout=idempotent.stdout,changed_refusal=denied.stderr,corrupt_refusal=corrupt.stderr,unsafe_tests=unsafe_results,all_preflight_no_partial_writes=True,input_hashes=before,manifest_sha256=sha((S/'ARCHIVE_RESUME2_EVIDENCE_MANIFEST.json').read_bytes()),restorer_sha256=sha((S/'restore_archive_resume2_evidence.py').read_bytes()),physical_calls=0)
(A/'STAGING_REPORT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items()if k not in ['exclusions','aliases','input_hashes','unsafe_tests','changed_refusal','corrupt_refusal']},indent=2))
