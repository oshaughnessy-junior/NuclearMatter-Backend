"""Stage archive_resume1 losslessly; no uploads, physical calls or upstream writes."""
from pathlib import Path
import json,hashlib,gzip,base64,shutil,subprocess,sys
B=Path('/Users/rossma/Genesis_LANL/outputs/nmb-finite-t-resume-20261017');A=B/'archive_resume1/final';S=A/'staged';PREFIX='research/finite_t_resume_20261017/'
assert not S.exists(),'Refuse to rebuild an existing frozen archive_resume1 stage'
S.mkdir(parents=True)
sha=lambda raw:hashlib.sha256(raw).hexdigest()
git=lambda raw:hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
roots=['pc_support','independent_review_preparation','review_pc_support','pc_figure']
author=B/roots[0];reviewprep=B/roots[1];oldroot=B.parent/'nmb-finite-t-20261015';original_manifest={'files':json.loads((author/'MANIFEST.json').read_text())['files_sha256']}
for rel,h in original_manifest['files'].items():assert sha((author/rel).read_bytes())==h,rel
mf=json.loads((reviewprep/'MANIFEST.json').read_text())
for rel,h in mf['files'].items():assert sha((reviewprep/rel).read_bytes())==h,rel
freeze=json.loads((author/'FREEZE.json').read_text());assert sha((author/'MANIFEST.json').read_bytes())==freeze['manifest_sha256']and sha((author/'conditional_pc.py').read_bytes())==freeze['candidate_sha256']
assert freeze['manifest_sha256']=='3323579161e95b79d378bfdb0d4692bbb0c281dfa12d0b7d6be65202cbaa9c7c'
assert freeze['candidate_sha256']=='5083e7857f3a9e256f4565b6ba19ce87f572b84abf48a381774f2bb364262e2f'
review=B/'review_pc_support';assert sha((review/'REVIEW.md').read_bytes())=='fe591a912ec7210b4f8897a3714b67c95299134d68efb76eb27409d917d0a7f6';assert sha((review/'MANIFEST.json').read_bytes())=='d6b5c2589145b64b1ac397eebc522e37d04b0eb21bd27cce66277aa684d94304';assert sha((review/'CHECKPOINT.json').read_bytes())=='7af4693def6ce472d17fe2692bd15cba7559fd1bb4c05c0860d420b1501a283f'
for rel,h in json.loads((review/'MANIFEST.json').read_text())['files_sha256'].items():assert sha((review/rel).read_bytes())==h
for rel,h in json.loads((B/'pc_figure/FREEZE.json').read_text())['files'].items():assert sha((B/'pc_figure'/rel).read_bytes())==h
oldpins=json.loads((author/'ORIGINAL_SOURCE_PIN.json').read_text());preppins=json.loads((reviewprep/'LOCK.json').read_text())['sourcepins']
for rel,h in oldpins.items():assert sha(Path(rel).read_bytes())==h,rel
for rel,h in preppins.items():assert sha((oldroot/rel).read_bytes())==h,rel
index={}
for i in range(1,49):
 root=oldroot/f'archive{i}/staged';path=root/f'ARCHIVE{i}_EVIDENCE_MANIFEST.json'
 if not path.exists():continue
 for rel,r in json.loads(path.read_text()).items():index.setdefault(r['original_sha256'],[]).append((i,root,rel,r))
pr54base=json.loads((oldroot/'production_weak_spectral_packaging_review/BASE_PIN.json').read_text())['files']
def resolve(h):
 if h in index:
  i,root,rel,r=index[h][0];packed=(root/r['stored_path']).read_bytes();assert sha(packed)==r['stored_sha256'];raw=gzip.decompress(packed)if r['encoding']=='gzip'else packed;assert sha(raw)==h
  return dict(original_campaign='nmb-finite-t-20261015',archive=i,original_path=rel,stored_path=r['stored_path'],sha256=h)
 for rel,r in pr54base.items():
  if r['sha256']==h:return dict(commit='c17472b8a84655380ab64973ef883a6cd5fb2346',source_path=rel,git_blob=r.get('git_blob'),sha256=h)
 return dict(sha256=h,resolution='Exactoriginalsourcehash retained; sourcepath pin is authoritative')
resource=B/'RESOURCE_AUDIT-20261017-1911.json';assert resource.is_file()
extra1=A/'SOURCE_RESOLUTION_LEDGER.json';extra1.write_text(json.dumps(dict(original_source_pins={p:resolve(h)for p,h in oldpins.items()},independent_prep_source_pins={p:resolve(h)for p,h in preppins.items()},expected_research_base_commit='17bea9d681a4c02a0757e8823fc213770767080a',expected_research_base_tree='53e39441c3d98ae0cfc61e8c0be24d20fbae1329',remote_head_lease='Root must refresh before publication',new_scientific_calls=0),indent=2)+'\n')
extra2=A/'ARCHIVER_RESOURCE_OBSERVATION.json';extra2.write_text(json.dumps(dict(utc='2026-10-17T19:12:27Z',account_id='82f4cfd9-9e20-47cf-9997-761328d89f60',limit_id='codex',window_minutes=10080,used_percent=92,reset_unix=1792652325,winddown=97,hard_stop=99,credits_consumed=False,identity_reset_verified=True,raw_other_agent_response_not_claimed=True),indent=2)+'\n')
extras=[extra1,extra2,resource,A.parent/'initial_builder_failure/initial_builder.py',A.parent/'initial_builder_failure/FAILURE.json',A.parent/'build_archive_resume1.py',A/'build_final.py',A.parent/'CHECKPOINT.json',A.parent/'STAGING_REPORT.json']
assessment=json.loads((author/'ASSESSMENT.json').read_text());assert len(assessment['checks'])==90 and len(assessment['forward_rows'])==2
counts={f:len(json.loads((author/f).read_text()).get('rows',[]))for f in ['CONVERGENCE_RAW.json','SEMI_INFINITE_RADIAL_RAW.json','PRE_LAGUERRE_CONVERGENCE_RAW.json']}

def input_inventory():
 paths=[p for root in roots for p in (B/root).rglob('*')if p.is_file()and '__pycache__'not in p.parts]
 paths +=[Path(rel)for rel in oldpins]+[oldroot/rel for rel in preppins]+extras
 return{str(p):sha(p.read_bytes())for p in paths}
before=input_inventory()
manifest={};exclusions=[];sources={};entries=[];binary=[];canonical={};aliases=[]
overlays={'package_copy/src/nmbackend/reference/finite_t_baryons.py','package_copy/src/nmbackend/reference/finite_t_leptons.py'}
for root in roots:
 for p in sorted((B/root).rglob('*')):
  if not p.is_file():continue
  rel=str(p.relative_to(B));local=str(p.relative_to(B/root));reason=None
  if root=='pc_figure'and p.suffix in ['.png','.pdf']:reason='Frozen image bytes reserved for papers; exact SHA in pc_figure/FREEZE.json; generator/data archived here'
  elif any(x in p.parts for x in ['__pycache__','.pytest_cache','.cache','.mplconfig']):reason='runtime/font cache'
  elif any(x in p.parts for x in ['baseline','test_mirror','candidate_package','package_base','package_copy','wheels','build','dist'])or p.suffix in ['.pyc','.so','.whl']:reason='full package/baseline/build/wheel copy; exact original SHA retained only'
  elif root==roots[0]and local.startswith('pr55_reference/')and sha(p.read_bytes())in oldpins.values():reason='ExactOct15PR55/prerequisitecopy; '+json.dumps(resolve(sha(p.read_bytes())),sort_keys=True)
  if reason:
   exclusions.append(dict(path=rel,reason=reason,sha256=sha(p.read_bytes())))
   continue
  raw=p.read_bytes();raw.decode('utf-8') # fail loudly for unexpected binary
  key=sha(raw)
  if key in canonical:
   record=canonical[key].copy();manifest[rel]=record;sources[rel]=p;aliases.append(dict(original=rel,canonical_stored=record['stored_path'],original_sha256=key));continue
  packed=len(raw)>40000 or(p.suffix=='.json'and len(raw)>20000)
  data=gzip.compress(raw,mtime=0)if packed else raw
  assert not packed or data==gzip.compress(raw,mtime=0)
  stored=rel+('.gz'if packed else'');dest=S/stored;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
  record=dict(encoding='gzip'if packed else'utf-8',stored_path=stored,original_sha256=key,stored_sha256=sha(data))
  canonical[key]=record;manifest[rel]=record;sources[rel]=p
for p in extras:
 rel=str(p.relative_to(B));raw=p.read_bytes();key=sha(raw)
 if key in canonical:record=canonical[key].copy();aliases.append(dict(original=rel,canonical_stored=record['stored_path'],original_sha256=key))
 else:
  packed=len(raw)>40000 or(p.suffix=='.json'and len(raw)>20000);data=gzip.compress(raw,mtime=0)if packed else raw;stored=rel+('.gz'if packed else'');dest=S/stored;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data);record=dict(encoding='gzip'if packed else'utf-8',stored_path=stored,original_sha256=key,stored_sha256=sha(data));canonical[key]=record
 manifest[rel]=record;sources[rel]=p
scope='''# Resumed archive1 FINAL: selected softPC alternate integral

DIFFERENT selected review67PASS0FAIL at author.1 and heldout.13 MeV; author.15 has no fresh independent integrated check. Review shares CM event and Laguerre radial family; literal old spinors, cap quadrature and unreduced flux normalization differ. This research alternate does not repair PR55. No wholeenergy qualification. Original
PR55positive-shell guard failure, incomingendpoint attempt, pre-Laguerre version,
finalsource and all unique raw phase/convergence/semiinfinite records retained.
Blindreview16oldsourcepins/analyticanchors/plan remain static preparation only.
Frozen review_pc_support admitted only after root exact hash handoff. Binder is excluded and its review unread. Frozen figure generator/data and image pins admitted; image bytes go only to papers. Initial pending archive checkpoint, helper fixture failure and report remain preserved.

ExactOct15PR55 namespace copies exclude only bytes matching originalsourcepins,
with sourcepath/hash/priorarchive or immutablePR54 resolution; unique initializer
and all newscience source/evidence remain. Runtimecachecopies excluded withhashes.
Resourceledger retains reported transient reset drift as a REPORT, clearly
separate from actual fresh root observations; no unavailable raw response is
invented. No resetcredit or policytolerance was used. Resource/account data stay
in backend metadata and are excluded from the planned scientific paper note.

CanonicalBYTEdeduplication retains every originalpath/hash. Detailed current,
freephysicalmasses/FD/spin/phase conventions unchanged; no coefficient is made
fromKMS, numericalwidth or falsezero. Semi-infinite quadrature comparison is not
a rigorous weighted-tail bound. No newphysics, imports, builds, installs,
GitHub/Git writes. Stopnewwork97/hardcheckpoint99 or identity/windowdrift.
Allpreflight saferestorer rejects corrupt/unsafe/changed output before ANYwrite.

Excluded files (original SHA256):

'''
scope+=''.join(f"- {e['path']} — {e['sha256']} — {e['reason']}\n"for e in exclusions)
(S/'ARCHIVE_RESUME1_SCOPE.md').write_text(scope)
(S/'ARCHIVE_RESUME1_EVIDENCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
restorer='''"""Verify/restore archive_resume1 evidence; refuse changed files and unsafe paths."""
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
    manifest=json.loads((ROOT/'ARCHIVE_RESUME1_EVIDENCE_MANIFEST.json').read_text())
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
(S/'restore_archive_resume1_evidence.py').write_text(restorer)
# Validate every original/stored round trip and then actual restoration in a
# disposable owned fixture; no frozen original or stage file is changed.
for rel,rec in manifest.items():
 stored=(S/rec['stored_path']).read_bytes();raw=gzip.decompress(stored)if rec['encoding']=='gzip'else stored
 assert raw==sources[rel].read_bytes()and sha(raw)==rec['original_sha256']
fixture=A/'roundtrip_fixture';shutil.copytree(S,fixture)
def restore(*args):return subprocess.run([sys.executable,'-B',str(fixture/'restore_archive_resume1_evidence.py'),*args],cwd=A,capture_output=True,text=True,timeout=30)
verify=restore('--verify-only');assert verify.returncode==0,verify.stderr
for rel,rec in manifest.items():
 if rec['encoding']=='gzip':assert not(fixture/rel).exists()
actual=restore();assert actual.returncode==0,actual.stderr
for rel,p in sources.items():assert(fixture/rel).read_bytes()==p.read_bytes()
idempotent=restore();assert idempotent.returncode==0,idempotent.stderr
def clean():
 shutil.rmtree(fixture);shutil.copytree(S,fixture)
def snap():return{str(p.relative_to(fixture)):sha(p.read_bytes())for p in fixture.rglob('*')if p.is_file()}
clean();mm=json.loads((fixture/'ARCHIVE_RESUME1_EVIDENCE_MANIFEST.json').read_text());witness='changed_output_alias_witness.txt';mm[witness]=next(iter(mm.values())).copy();(fixture/'ARCHIVE_RESUME1_EVIDENCE_MANIFEST.json').write_text(json.dumps(mm));changed=fixture/witness;changed.write_bytes(b'changed evidence must survive\n');old=snap();denied=restore();assert denied.returncode!=0 and'Refusing to overwrite changed evidence'in denied.stderr and snap()==old
clean();rec=next(iter(manifest.values()));packed=fixture/rec['stored_path'];packed.write_bytes(packed.read_bytes()+b'poison');old=snap();corrupt=restore();assert corrupt.returncode!=0 and'Stored checksum mismatch'in corrupt.stderr and snap()==old
unsafe_results=[]
for original,stored in [('../archive_resume1_escape_witness.txt',None),('/tmp/archive_resume1_escape_witness.txt',None),('valid_unsafe_source_witness','../source_escape.txt')]:
 clean();mm=json.loads((fixture/'ARCHIVE_RESUME1_EVIDENCE_MANIFEST.json').read_text());rr=next(iter(mm.values())).copy()
 if stored:rr['stored_path']=stored
 mm[original]=rr;(fixture/'ARCHIVE_RESUME1_EVIDENCE_MANIFEST.json').write_text(json.dumps(mm));old=snap();bad=restore();assert bad.returncode!=0 and'Unsafe archive-relative path'in bad.stderr and snap()==old
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
coverage={rel:('archived'if roots[0]+'/'+rel in manifest else'excluded-with-hash')for rel in original_manifest['files']}
result=dict(status='LOCAL_STAGE_VERIFIED_NO_UPLOAD',author_manifest_paths_verified=len(original_manifest['files']),original_source_pins_verified=len(oldpins),blind_preparation_source_pins_verified=len(preppins),author_selected_checks=90,selected_energies=[.1,.15],raw_profile_counts=counts,independent_review='ACCEPTED_SELECTED_67PASS0FAIL',active_binder='EXCLUDED_UNREAD',qualification='SELECTED_RESEARCH_ALTERNATE_NOT_PR55_REPAIR_NO_WHOLEENERGY_OR_TAILBOUND_OR_TRANSPORT',manifest_covered_archived=sum(v=='archived'for v in coverage.values()),manifest_covered_excluded=sum(v!='archived'for v in coverage.values()),archived_originals=len(manifest),author_originals=sum(p.startswith(roots[0]+'/')for p in manifest),review_originals=sum(p.startswith(roots[1]+'/')for p in manifest),excluded_files=exclusions,canonical_unique_original_payloads=len(canonical),canonical_aliases=aliases,unsafe_path_tests=unsafe_results,all_preflight_tests_no_partial_writes=True,text_entries=len(entries),compressed_binary_entries=len(binary),payload_batches=len(batches),total_staged_blobs=len(localhash),roundtrip_originals_verified=len(manifest),verify_only_stdout=verify.stdout,restore_stdout=actual.stdout,idempotent_stdout=idempotent.stdout,changed_file_refusal_stderr=denied.stderr,corrupt_stored_refusal_stderr=corrupt.stderr,all_original_bytes_unchanged=True,hot_beta_results_sha256=sha((oldroot/'hot_beta_moments/results.json').read_bytes()),bootstrap_manifest_sha256=sha((S/'ARCHIVE_RESUME1_EVIDENCE_MANIFEST.json').read_bytes()),restorer_sha256=sha((S/'restore_archive_resume1_evidence.py').read_bytes()),scope_sha256=sha((S/'ARCHIVE_RESUME1_SCOPE.md').read_bytes()),input_hashes=before)
(A/'STAGING_REPORT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items()if k not in ['input_hashes','excluded_files','changed_file_refusal_stderr','corrupt_stored_refusal_stderr','canonical_aliases','unsafe_path_tests']},indent=2))
