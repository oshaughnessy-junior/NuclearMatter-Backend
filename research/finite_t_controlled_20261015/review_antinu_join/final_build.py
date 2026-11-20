"""Final accepted-manifest build and private valid-fixture unit/pass guards."""
from pathlib import Path
import importlib.util,json,hashlib,copy
R=Path(__file__).resolve().parents[1];H=Path(__file__).parent
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
J=load(R/'antinu_source_join/join.py','final_join');checks=[];rows=[]
def gate(name,ok,detail=None):checks.append(dict(name=name,passed=bool(ok),detail=detail))
def reject(name,fn,reason):
 try:fn();message='accepted';good=False
 except ValueError as e:message=str(e);good=reason in message
 gate(name,good,message)
# Different source encodings of exact missing incoming occupation are valid.
for reason in ['empty_support','absent_initial_antinu_occupation']:
 s=dict(x=dict(y=dict(value=0.,log=None,representable=True,reason=reason)))
 gate('physical exact zero '+reason,J.value(s,'x','y')==0)
for name,value,log,reason in [('positive floor',0.,-1000.,None),('positive missing log',1.,None,None),('positive infinite log',1.,float('inf'),None),('unexplained zero',0.,None,'unknown')]:
 s=dict(x=dict(y=dict(value=value,log=log,representable=True,reason=reason)))
 reject(name,lambda s=s:J.value(s,'x','y'),'log'if name!='unexplained zero'else'zero floor')
manifest_path=R/'antinu_source_join/ACCEPTED_MANIFEST.json'
if not manifest_path.exists():raise RuntimeError('Accepted manifest not available; run this final gate only after source reviews freeze')
manifest=json.loads(manifest_path.read_text());J.verify_manifest(manifest);out=J.build();rows.append(dict(actual_build=out));gate('actual accepted manifest full build',out['status']=='complete')
gate('both independent review snapshots present',all((R/n).exists()for n in ['review_hot_beta/REVIEW.md','review_hot_beta/CHECKPOINT.json','review_pc_moments/REVIEW.md','review_pc_moments/MANIFEST.json']))
# Private verbatim copy, with all mandatory hash-covered sources copied.
base=H/'final_synthetic_fixture';dest=base/'antinu_source_join';dest.mkdir(parents=True,exist_ok=True)
for name in manifest['sha256']:
 target=base/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((R/name).read_bytes())
(dest/'join.py').write_bytes((R/'antinu_source_join/join.py').read_bytes())
(dest/'ACCEPTED_MANIFEST.json').write_text(json.dumps(manifest));S=load(dest/'join.py','private_final_join')
gate('private fixture valid baseline',S.build()['status']=='complete')
pr=base/'positron_capture_moments/refinement.json';original=pr.read_bytes();mcopy=copy.deepcopy(manifest)
d=json.loads(original)
for row in d['rows']:
 if row.get('kind')=='total'and row['result']['controls']['initial_order']==12:row['result']['units']='WRONG UNITS'
pr.write_text(json.dumps(d));mcopy['sha256']['positron_capture_moments/refinement.json']=hashlib.sha256(pr.read_bytes()).hexdigest();(dest/'ACCEPTED_MANIFEST.json').write_text(json.dumps(mcopy))
reject('unit mismatch despite private matching hash',S.build,'Moment unit conventions differ')
d=json.loads(original);d['passed']=False;pr.write_text(json.dumps(d));mcopy['sha256']['positron_capture_moments/refinement.json']=hashlib.sha256(pr.read_bytes()).hexdigest();(dest/'ACCEPTED_MANIFEST.json').write_text(json.dumps(mcopy))
reject('failed coarse report despite private matching hash',S.build,'Qualified report must pass')
(H/'final_build_results.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256((R/'antinu_source_join/join.py').read_bytes()).hexdigest(),checks=checks,rows=rows,all_pass=all(c['passed']for c in checks)),indent=2)+'\n')
print(json.dumps(dict(checks=len(checks),failed=[c for c in checks if not c['passed']])))
