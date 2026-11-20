from pathlib import Path
import shutil,json,hashlib,ast
P=Path('/Users/rossma/Genesis_LANL/outputs/nmb-finite-t-20261015/production_weak_packaging_review');A=P.parent/'production_weak_port';S=P/'pre_numerical_acceptance';S.mkdir();old=json.loads((P/'PREPARATION.json').read_text());oldscript=(P/'prepare.py').read_text()
for p in list(P.iterdir()):
 if p!=S:shutil.move(str(p),str(S/p.name))
new=json.loads((A/'CANDIDATE_MANIFEST.json').read_text());assert set(old['candidate_paths'])==set(new['files']);changed=[p for p,h in old['candidate_paths'].items()if new['files'][p]!=h];assert set(changed)=={'src/nmbackend/reference/_weak_quadrature.py','tests/test_weak_reference_moments.py'}
def masked(p):
 t=ast.parse(p.read_text())
 for n in t.body:
  if isinstance(n,ast.ClassDef)and n.name=='SignedQuantity':n.body=[q for q in n.body if not(isinstance(q,ast.FunctionDef)and q.name=='__post_init__')]
 return ast.dump(t,include_attributes=False)
assert masked(S/'package_copy/src/nmbackend/reference/_weak_quadrature.py')==masked(A/'staged/src/nmbackend/reference/_weak_quadrature.py')
s=oldscript.replace('795975d51e75e47360822e70845b1eac5ba68df2f134877b7ed1e34ca339789d','80f0b4207003353b8bf59f1357c28baedbf85ccb06d0a85b9289bceb798a174e').replace('12404beabd3aaf98db37cb2c880ea061cc5d55edd1e813bf8ab8a9c162f11647','347ee290afa7eb4d177b92c2ee65056c90cea2a388d13d74ac4e5f5e8f8c6cd7').replace('PREPARATION_ONLY_NOT_ACCEPTANCE_WAITING_NUMERICAL_SIGNAL','REPAIRED_PACKAGE_PREPARED_NUMERICAL_ACCEPTANCE_RECEIVED').replace('WAITING','ROOT_SIGNAL_RECEIVED')
(P/'prepare.py').write_text(s)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
R=P.parent/'review_production_weak_port';assert sha(R/'REVIEW.md')=='ad05069c9ed312408f4b18189358547f3f24645dfd7123b458d029e1a66d2cc3';assert sha(R/'MANIFEST.json')=='fc8e4d031ba7a36f0145e15f1fbe5bf5112306bece0ce861a23e0a17b891dc22'
proof=dict(old_candidate='795975d51e75e47360822e70845b1eac5ba68df2f134877b7ed1e34ca339789d',new_candidate='80f0b4207003353b8bf59f1357c28baedbf85ccb06d0a85b9289bceb798a174e',changed_paths=changed,all_other_nine_overlays_identical=True,helper_AST_identical_except_SignedQuantity_post_init=True,initial_preparation_preserved={str(p.relative_to(S)):sha(p)for p in S.rglob('*')if p.is_file()},accepted_review=sha(R/'REVIEW.md'),accepted_review_manifest=sha(R/'MANIFEST.json'))
(P/'REFRESH_PROOF.json').write_text(json.dumps(proof,indent=2)+'\n')
