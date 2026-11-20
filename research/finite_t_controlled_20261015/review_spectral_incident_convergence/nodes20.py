from pathlib import Path
import json,hashlib,sys,dataclasses,time
P=Path(__file__).resolve().parent;B=P.parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();L=json.loads((P/'LOCK.json').read_text());ch=sys.argv[1];first=int(sys.argv[2]);assert ch in L['rules']and first in range(0,20,4)
def verify():
 for n,h in L['sourcepins'].items():assert sha(B/n)==h,n
 for n,h in L['study_frozen_hashes'].items():assert sha(B/'spectral_incident_convergence'/n)==h,n
verify();sys.dont_write_bytecode=True;sys.path.insert(0,str(B/L['source_package']/'src'))
from nmbackend.reference import weak_reference as W
assert Path(W.__file__).resolve()==(B/L['source_package']/'src/nmbackend/reference/weak_reference.py').resolve();bath=W.FreeElectronBath(**L['bath']);controls=W.SpectralControls(**L['controls']);r=L['rules'][ch];out=P/'raw'/f'{ch}_{first:02d}.json';assert not out.exists();start=time.monotonic()
d=dict(channel=ch,first=first,stop=first+4,order=20,rule=r,bath=L['bath'],controls=L['controls'],lock_sha256=sha(P/'LOCK.json'),worker_sha256=sha(Path(__file__)),sourcepins=L['sourcepins'],complete=False,rows=[])
def save():d['seconds']=time.monotonic()-start;out.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
save()
for i in range(first,first+4):
 verify();z=W.spectral_coefficients(ch,bath,r['nodes'][i],controls);verify();d['rows'].append(dict(index=i,energy=r['nodes'][i],weight=r['weights'][i],result=dataclasses.asdict(z)));save()
d['complete']=True;save();print(ch,first,'4complete',d['seconds'])
