"""Apply accepted frozen11path overlay to previously frozen realPR54base."""
from pathlib import Path
import json,hashlib,shutil,ast
P=Path(__file__).resolve().parent;A=P.parent/'production_weak_spectral_port';R=P.parent/'review_production_weak_spectral_port';B=P/'package_copy';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not B.exists()
initial=P/'pre_numerical_acceptance';initial.mkdir()
for p in P.iterdir():
 if p.is_file()and p.name!='prepare_candidate.py':shutil.copy2(p,initial/p.name)
base=json.loads((P/'BASE_PIN.json').read_text());assert base['head']=='c17472b8a84655380ab64973ef883a6cd5fb2346'and len(base['files'])==86
assert all(sha(P/'package_base'/p)==r['sha256']for p,r in base['files'].items())
assert sha(A/'CANDIDATE_MANIFEST.json')=='dcfbafe85645caa4ac6f2ea1f4cd3a7cb55775d030eddc0c9c5a8c9211a8b216';assert sha(A/'MANIFEST.json')=='f90bb7d3ee0bbd0ae6025f68edb8ed14b431a5bbcf7a0d7bcf5fa16c2c5ed78c'
for f,h in [('REVIEW.md','89cd023016062705490e1a58dcc88ff1fbd757a11a7cc936e2be7213acf4f096'),('MANIFEST.json','9899fb490605039de0731f6dccf54451fd7ca4ef373aad298db6b338cf861096'),('REPAIR_RECHECK.md','2d760af394f96a523599ce00da6b5492615c2e5750f14141d69d48723866a9fa')]:assert sha(R/f)==h,f
m=json.loads((A/'CANDIDATE_MANIFEST.json').read_text());assert m['base_commit']==base['head']and len(m['files'])==11 and len(m['modified'])==4 and len(m['new'])==7
allowed={'README.md','docs/weak_reference.md','docs/weak_spectral_validation.md','examples/free_weak_reference.py','src/nmbackend/reference/weak_reference.py','src/nmbackend/reference/_weak_spectral_common.py','src/nmbackend/reference/_weak_spectral_capture.py','src/nmbackend/reference/_weak_spectral_decay.py','tests/data/free_weak_spectral_reference.json','tests/test_weak_reference_spectra.py','tests/test_weak_reference_reconstruction.py'};assert set(m['files'])==allowed
shutil.copytree(P/'package_base',B)
for p,h in m['files'].items():
 assert sha(A/'staged'/p)==h;q=B/p;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(A/'staged'/p,q)
unchanged={p:r for p,r in base['files'].items()if p not in m['modified']};assert all(sha(B/p)==r['sha256']for p,r in unchanged.items())
for p,h in json.loads((A/'MANIFEST.json').read_text()).get('sha256',{}).items():assert sha(A/p)==h,p
apins={str(p.relative_to(A)):sha(p)for p in A.rglob('*')if p.is_file()and '__pycache__'not in p.parts}
rpins={str(p.relative_to(R)):sha(p)for p in R.rglob('*')if p.is_file()and '__pycache__'not in p.parts}
static={}
for p in m['files']:
 if p.endswith('.py'):
  text=(B/p).read_text();tree=ast.parse(text,feature_version=(3,9));static[p]=dict(python39_grammar=True,imports=[ast.unparse(n)for n in ast.walk(tree)if isinstance(n,(ast.Import,ast.ImportFrom))])
(P/'BEFORE_PIN.json').write_text(json.dumps(dict(author=apins,numeric_review=rpins),indent=2)+'\n')
(P/'CANDIDATE_PREPARATION.json').write_text(json.dumps(dict(base_commit=base['head'],candidate_manifest=sha(A/'CANDIDATE_MANIFEST.json'),candidate_paths=m['files'],modified=m['modified'],new=m['new'],unchanged_base_files=unchanged,static=static,accepted_numeric_review=sha(R/'REVIEW.md'),accepted_repair_review=sha(R/'REPAIR_RECHECK.md'),baseline_pin_preserved=sha(initial/'BASE_PIN.json')),indent=2)+'\n')
print('Accepted11pathoverlay;86originalbasefiles preserved,82unchangedcopyfiles; no imports/calls yet')
