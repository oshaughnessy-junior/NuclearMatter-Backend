from pathlib import Path
import json,hashlib,shutil,ast
P=Path(__file__).resolve().parent;R=P.parent;A=R/'production_weak_spectral_port';h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
expect={'CANDIDATE_MANIFEST.json':'18da0dfe7f52c8fea7c59d9fe7e1687b5893738c36cf836091587cff02406ad0','MANIFEST.json':'c86494e2841b534c28b1690ff34fc3ae925f0409726967ac1d3c7afb975573b3','AUTHOR_REPORT.md':'f338ea879c2f3ed062868b5b4251a45370f6e4221ac3f08930205d73bc343257'}
for n,v in expect.items():assert h(A/n)==v,n
m=json.loads((A/'CANDIDATE_MANIFEST.json').read_text());assert len(m['files'])==11 and len(m['modified'])==4 and len(m['new'])==7
M=json.loads((A/'MANIFEST.json').read_text());allfiles=M.get('sha256',M.get('files'));assert isinstance(allfiles,dict)
for n,v in allfiles.items():assert h(A/n)==v,n
for n,v in m['files'].items():
 for b in['staged','test_mirror']:assert h(A/b/n)==v,(b,n)
old=json.loads((R/'production_weak_port/CANDIDATE_MANIFEST.json').read_text())['files'];unchanged={n:v for n,v in old.items()if n not in m['modified']}
for n,v in unchanged.items():assert h(A/'test_mirror'/n)==v,n
# EntireStageAtotal/classdefinitions unchanged, apart from explicitoldstub extension.
f='src/nmbackend/reference/weak_reference.py';t1,t2=[ast.parse(p.read_text())for p in[R/'production_weak_port/staged'/f,A/'staged'/f]];names=[n.name for n in t1.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))and n.name!='spectral_coefficients'];o={n.name:ast.dump(n,include_attributes=False)for n in t1.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))};a={n.name:ast.dump(n,include_attributes=False)for n in t2.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))};assert all(o[n]==a[n]for n in names)
for n,v in json.loads((P/'PREPARATION_MANIFEST.json').read_text())['files'].items():assert h(P/n)==v,n
shutil.copytree(A/'test_mirror',P/'candidate_copy',ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
for n in expect:shutil.copyfile(A/n,P/('PINNED_'+n))
out=dict(author_files=allfiles,candidate_files=m['files'],stageA_unchanged_files=unchanged,unchanged_stageA_AST_names=names,base_commit=m['base_commit'],prep_unchanged=True,candidate_not_imported_yet=True);(P/'CANDIDATE_PIN.json').write_text(json.dumps(out,indent=2)+'\n');print('PINNED',len(allfiles),'authorfiles',len(unchanged),'StageAfiles',len(names),'unchangedStageAdefinitions')
