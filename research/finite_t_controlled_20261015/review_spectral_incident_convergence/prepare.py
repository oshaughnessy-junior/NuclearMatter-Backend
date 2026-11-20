from pathlib import Path
import json,hashlib,datetime
import numpy as np
P=Path(__file__).resolve().parent;B=P.parent;A=B/'spectral_incident_convergence';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
expected={'REPORT_SUCCESSOR.md':'bfb5b999dcd5ca23f5d526c1259f5c8cebfb847af101934c061deafc981e1d02','MANIFEST_SUCCESSOR.json':'eab2820f61e8934a232a00f46e7fd46160b1ce8bee5161ee9a39e93255c789c4','CHECKPOINT_SUCCESSOR.json':'7a5d65d9c43431695290233887971fa10ab0fec045831bc73e15b967ba732a49','CANONICAL_FINITE_RANGE_RESULTS.json':'4ecd574c54174b461726891971a1b0b7fa597d134a6d27297f3fcc8fb940a22c'}
for n,h in expected.items():assert sha(A/n)==h,n
m=json.loads((A/'MANIFEST_SUCCESSOR.json').read_text())['sha256'];assert all(sha(A/n)==h for n,h in m.items());L=json.loads((A/'LOCK.json').read_text());source=B/'review_production_weak_spectral_port/repaired_copy';candidate=json.loads((B/'production_weak_spectral_port/CANDIDATE_MANIFEST.json').read_text());assert sha(B/'production_weak_spectral_port/CANDIDATE_MANIFEST.json')=='dcfbafe85645caa4ac6f2ea1f4cd3a7cb55775d030eddc0c9c5a8c9211a8b216'
for n,h in candidate['files'].items():assert sha(source/n)==h,n
packagepins={str(f.relative_to(B)):sha(f)for f in(source/'src/nmbackend/reference').glob('*.py')};rules={}
for ch,(lo,hi)in L['channels'].items():
 x,w=np.polynomial.legendre.leggauss(20);rules[ch]=dict(nodes=[float(v)for v in lo+(x+1)*(hi-lo)/2],weights=[float(v)for v in w*(hi-lo)/2],lower=lo,upper=hi)
lock=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),before_new_physical_calls=True,scope='DIFFERENT finite-incident integration review only; shared reviewed inner kernel NOT independent physical model',study_frozen_hashes=expected,study_manifest_files=m,bath=L['bath'],controls=L['controls'],radiation=L['radiation'],rules=rules,incident_order=20,chunk_nodes=4,timeout_seconds=119,positive_branch_relative_gate=3e-6,gross_LTE_gate=2e-9,entropy_affinity_gate=3e-9,aggregate_arithmetic_gate=3e-12,compared_reference_order=32,source_package=str(source.relative_to(B)),sourcepins=packagepins,stop_on_failed_selected_new_rule=True,no_rerung_without_root_direction=True,whole_energy_tails_OPEN=True,source_execution='new repaired dcfb independent20rule; old168ECPC+84successorbeta preserved separately')
assert not(P/'LOCK.json').exists();(P/'LOCK.json').write_text(json.dumps(lock,indent=2,allow_nan=False)+'\n');(P/'raw').mkdir();print('LOCK',sha(P/'LOCK.json'),'studyfiles',len(m),'no physicalcalls')
