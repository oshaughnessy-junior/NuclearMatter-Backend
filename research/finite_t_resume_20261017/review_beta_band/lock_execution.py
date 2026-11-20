from pathlib import Path
import hashlib,json,datetime
P=Path(__file__).resolve().parent;R=P.parent;B=R/'beta_band_review_preparation';A=R/'beta_band_refinement';h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();prep=json.loads((B/'LOCK.json').read_text());mf=json.loads((A/'MANIFEST.json').read_text());old=json.loads((R/'review_beta_support/LOCK.json').read_text())
assert h(B/'CHECKPOINT.json')=='57b5f60b328087485a31f9e176a6b02d406ad51338512797c98104b411f4931c';assert h(A/'MANIFEST.json')=='8ca14f16d1f013c2dfab9f4e718decd1e0b6ba04453973d472fbdecb57f62fd2'
assert all(h(A/f)==v for f,v in mf['files_sha256'].items());assert all(h(Path(f))==v for f,v in prep['source_pins'].items())
jobs=[]
for node in prep['incident_rule']['nodes']:jobs.append(dict(kind='oracle',E=node['E_MeV'],parent=24,rest_nu=24,electron=24,phi=12,node_index=node['index'],weight_MeV=node['weight_MeV'],role='incident24'))
for chk in prep['inner_checks']:
 node=prep['incident_rule']['nodes'][chk['index']];jobs.append(dict(kind='oracle',E=node['E_MeV'],parent=chk['parent'],rest_nu=chk['rest_nu'],electron=chk['electron'],phi=chk['phi'],node_index=chk['index'],role=chk['name']))
L={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'prep_sha256':h(B/'CHECKPOINT.json'),'prep_files':json.loads((B/'MANIFEST.json').read_text())['files_sha256'],'band_author_manifest':h(A/'MANIFEST.json'),'band_author_files':mf['files_sha256'],'author_files':old['author_files'],'jobs':jobs,'max_calls':28,'gate':3e-6}
(P/'LOCK.json').write_text(json.dumps(L,indent=2)+'\n')
text=(R/'review_beta_support/run_job.py').read_text();text=text.replace("verify();start=time.monotonic()","\nfor f,s in L['band_author_files'].items():assert sha(R/'beta_band_refinement'/f)==s,f\nverify();start=time.monotonic()")
(P/'run_job.py').write_text(text);(P/'bounded_job.py').write_bytes((R/'review_beta_support/bounded_job.py').read_bytes());print('LOCK',h(P/'LOCK.json'),'jobs',len(jobs))
