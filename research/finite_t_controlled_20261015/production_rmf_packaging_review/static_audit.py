from pathlib import Path
import ast,json,difflib
P=Path(__file__).resolve().parent;B=P/'package_copy';checks=[];calls={};imports={}
def ck(n,v):checks.append(dict(name=n,passed=bool(v)))
m=json.loads((P/'PREPARATION.json').read_text())['candidate_paths']
for path in m:
 if not path.endswith('.py'):continue
 raw=(B/path).read_text();tree=ast.parse(raw,feature_version=(3,9));ck('3.9 grammar '+path,True)
 imports[path]=[ast.unparse(n)for n in ast.walk(tree)if isinstance(n,(ast.Import,ast.ImportFrom))]
 calls[path]=sorted({ast.unparse(n.func)for n in ast.walk(tree)if isinstance(n,ast.Call)})
 ck('no post-minimum helper '+path,not any(x in raw for x in ['log_expit','np.trapezoid','slots=True','kw_only=True','strict=True','tomllib','scipy.differentiate']))
 for n in ast.walk(tree):
  if isinstance(n,ast.Call)and ast.unparse(n.func)=='quad_vec':ck('quad_vec no 1.8 args keyword',all(k.arg!='args'for k in n.keywords))
readme=(B/'README.md').read_text();old=(P.parent/'production_rmf_port/evidence/baseline_README.md').read_text();ck('README exact additive suffix',readme.startswith(old)and len(readme)>len(old))
(P/'README.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),readme.splitlines(True),fromfile='ffb baseline',tofile='candidate')))
for path in ['src/nmbackend/reference/finite_t_rmf.py','src/nmbackend/reference/_rmf_response.py','src/nmbackend/reference/_rmf_moments.py']:
 ck('no campaign runtime path '+path,not any(t in(B/path).read_text()for t in ['Genesis_LANL','outputs/','sys.path','importlib.util','production_lepton','production_baryon']))
fixture=json.loads((B/'tests/data/finite_t_gm1_reference.json').read_text());ck('fixture explicitly INTERNAL only',fixture['no_physics_executed']and len(fixture['rows'])==3 and all(r['external_code_executed']is False for r in fixture['rows']))
ck('only README overlays existing baseline outside new paths',len(m)==9 and len([p for p in m if p=='README.md'])==1)
out=dict(passed=sum(x['passed']for x in checks),failed=[x for x in checks if not x['passed']],checks=checks,imports=imports,calls=calls,scope='STATIC grammar/API inspection only, not Python3.9 or SciPy1.7 execution',version_sources=['https://docs.scipy.org/doc/scipy-1.7.0/reference/generated/scipy.integrate.quad_vec.html','https://numpy.org/doc/1.21/numpy-ref.pdf'],version_evidence='Official SciPy1.7 generated index exposes quad_vec. Direct historical HTML and raw source requests failed cache lookup; retained as limitation. Existing quad_vec max norm/full_output use predates1.7; 1.8-only args keyword not used. NumPy1.21 reference documents logaddexp. Other used NumPy/SciPy primitives are longstanding. No minimum runtime executed.')
(P/'static_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out['passed'],out['failed'])
