"""Resume ONLY missing actual spectral nodes after preserved quota-metadata hold."""
from pathlib import Path
import importlib.util,sys,json,time,shutil
P=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('hotPC_owned_resume_total',P/'total.py');m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
c=m.contract('fine');f=P/'total_fine_3_partial.json';d=json.loads(f.read_text());rows=d['rows'];out=P/'total_fine_3.json'
assert not out.exists()and d['qualified']is False and d['contract']==c and [r['index']for r in rows]==list(range(48,48+len(rows)))
assert len(rows)==5
for r in rows:assert r['E']==c['nodes'][r['index']]and r['weight']==c['weights'][r['index']]and r['spectrum']['implementation_sha256']==c['spectrum_sha']and r['spectrum']['source_pins']==c['sources']
shutil.copyfile(f,P/'interrupted_total_fine3_partial.json');start=time.monotonic();n,a,e,ph,t,ot=c['controls']
for i in range(48+len(rows),64):
 if time.monotonic()-start>100:raise TimeoutError('boundedexactmissingnode resume')
 r=m.S.integrate(c['nodes'][i],lepton_order=n,angle_order=a,energy_order=e,phi_order=ph,tail=t,outer_tail=ot,budget_seconds=15.);rows.append(dict(index=i,E=c['nodes'][i],weight=c['weights'][i],spectrum=r));f.write_text(json.dumps(dict(contract=c,rows=rows,qualified=False),indent=2)+'\n')
assert m.contract('fine')==c and [r['index']for r in rows]==list(range(48,64))
out.write_text(json.dumps(dict(contract=c,rows=rows,qualified=True,seconds=sum(r['spectrum']['seconds']for r in rows),resume=dict(completed_beforehold=5,newlycomputed=11,seconds=time.monotonic()-start)),indent=2)+'\n');print('fine3 complete resume',time.monotonic()-start)
