from pathlib import Path
import json,hashlib,sys,time,dataclasses
P=Path(__file__).resolve().parent;B=P.parent;L=json.loads((P/'LOCK.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
 for n,h in L['sourcepins'].items():assert sha(B/n)==h,n
verify();sys.dont_write_bytecode=True;sys.path.insert(0,str(B/'review_production_weak_spectral_port/repaired_copy/src'))
from nmbackend.reference import weak_reference as W
assert sha(Path(W.__file__))==L['sourcepins']['production_weak_spectral_port/test_mirror/src/nmbackend/reference/weak_reference.py'];b=W.FreeElectronBath(**L['bath']);rows=[]
for probe in L['preregistered_possible_calls']:
 verify();start=time.monotonic()
 try:r=W.spectral_coefficients(probe['channel'],b,probe['energy_MeV'],W.SpectralControls(**probe['controls']));z=dict(input=probe,completed=True,result=dataclasses.asdict(r))
 except Exception as e:z=dict(input=probe,completed=False,exception=dict(type=type(e).__name__,message=str(e)))
 verify();z['seconds']=time.monotonic()-start;rows.append(z);(P/'WITNESS_RESULTS.json').write_text(json.dumps(dict(rows=rows,sourcepins=L['sourcepins'],complete=len(rows)==3,physical_calls=len(rows)),indent=2,allow_nan=False)+'\n');print(probe['channel'],probe['controls']['tail'],z['completed'],z.get('exception'),z['seconds'],flush=True)
