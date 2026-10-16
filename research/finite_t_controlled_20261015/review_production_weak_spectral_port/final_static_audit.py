from pathlib import Path
import hashlib,json,ast
P=Path(__file__).resolve().parent;A=P.parent/'production_weak_spectral_port';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();checks=[]
def ck(n,ok):checks.append(dict(name=n,passed=bool(ok)))
for key in ['MANIFEST.json','CANDIDATE_MANIFEST.json']:
 expected={'MANIFEST.json':'f90bb7d3ee0bbd0ae6025f68edb8ed14b431a5bbcf7a0d7bcf5fa16c2c5ed78c','CANDIDATE_MANIFEST.json':'dcfbafe85645caa4ac6f2ea1f4cd3a7cb55775d030eddc0c9c5a8c9211a8b216'};ck('finalfrozenfingerprint '+key,sha(A/key)==expected[key])
for n,h in json.loads((A/'MANIFEST.json').read_text())['sha256'].items():assert sha(A/n)==h,n
ck('all1113frozenauthorfilesunchanged',True)
for n,h in json.loads((P/'INITIAL_REVIEW_MANIFEST.json').read_text())['files'].items():assert sha(P/n)==h,n
ck('all216initialreviewfilesunchanged',True)
for n,h in json.loads((A/'CANDIDATE_MANIFEST.json').read_text())['files'].items():ck('exactrepairedcopy '+n,sha(P/'repaired_copy'/n)==h)
old=ast.parse((P/'candidate_copy/src/nmbackend/reference/_weak_spectral_common.py').read_text());new=ast.parse((P/'repaired_copy/src/nmbackend/reference/_weak_spectral_common.py').read_text());o=next(n for n in old.body if isinstance(n,ast.FunctionDef)and n.name=='reconstruct');nn=next(n for n in new.body if isinstance(n,ast.FunctionDef)and n.name=='_assemble_spectral_nodes');begin=lambda ns:next(i for i,n in enumerate(ns)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='f'for t in n.targets));dump=lambda ns:[ast.dump(n,include_attributes=False)for n in ns]
ck('finalarithmeticAST exactlymoved',dump(o.body[begin(o.body):-1])==dump(nn.body[begin(nn.body):-1]))
z=dict(passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],checks=checks,physical_calls=0);(P/'FINAL_STATIC_AUDIT.json').write_text(json.dumps(z,indent=2)+'\n');print(z['passed'],'PASS',len(z['failed']),'FAIL');assert not z['failed']
