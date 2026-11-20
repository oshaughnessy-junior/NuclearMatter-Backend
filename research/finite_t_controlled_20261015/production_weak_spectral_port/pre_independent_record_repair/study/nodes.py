"""Owned actual-node harness. Candidate is read-only and acceptance pending."""
from pathlib import Path
import sys,json,hashlib,dataclasses,time
R=Path(__file__).resolve().parent;ROOT=R.parent
LOCK=json.loads((R/'LOCK.json').read_text());PINS=json.loads((R/'SOURCE_PIN.json').read_text())
H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
 for p,h in PINS.items():
  if H(ROOT/p)!=h:raise RuntimeError('SOURCE CHANGE STOP: '+p)
verify()
sys.path.insert(0,str(ROOT/'production_weak_spectral_port/test_mirror/src'))
from nmbackend.reference import weak_reference as W
assert Path(W.__file__).resolve()==(ROOT/'production_weak_spectral_port/test_mirror/src/nmbackend/reference/weak_reference.py').resolve()
ch=sys.argv[1];order=int(sys.argv[2]);startindex=int(sys.argv[3])
if ch not in LOCK['channels']or order not in LOCK['incident_orders']or startindex<0 or startindex>=order or startindex%LOCK['resources']['chunk_nodes']:raise ValueError('Only exactlocked jobs')
lo,hi=LOCK['channels'][ch];rule=W.EnergyRule.gauss(lo,hi,order);b=W.FreeElectronBath(**LOCK['bath']);c=W.SpectralControls(**LOCK['controls'])
path=R/'raw'/f'{ch}_{order}_{startindex:02d}.json'
if path.exists():raise RuntimeError('Refuse overwrite of existing actual node job')
stop=min(order,startindex+LOCK['resources']['chunk_nodes']);begin=time.monotonic()
out=dict(kind='actual detailed spectral nodes; candidate qualification pending',channel=ch,order=order,first_index=startindex,last_index_exclusive=stop,rule=dataclasses.asdict(rule),bath=dataclasses.asdict(b),controls=dataclasses.asdict(c),source_pin_sha256=H(R/'SOURCE_PIN.json'),lock_sha256=H(R/'LOCK.json'),worker_sha256=H(Path(__file__)),candidate_manifest_sha256=LOCK['candidate_manifest_sha256'],rows=[],complete=False)
def save():path.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
save()
try:
 for i in range(startindex,stop):
  verify();t=time.monotonic();node=W.spectral_coefficients(ch,b,rule.nodes[i],c)
  if not node.complete or node.j.log is None or node.true_chi.log is None:raise RuntimeError('Incomplete/falsezero integrated coefficient')
  if node.channel!=ch or node.bath!=b or node.controls!=c or node.energy!=rule.nodes[i]or node.differential_coordinates is not None:raise RuntimeError('Actual node identity mismatch')
  node.__post_init__();verify()
  out['rows'].append(dict(index=i,energy=rule.nodes[i],weight=rule.weights[i],result=dataclasses.asdict(node),KMS_log_residual=node.kms_log_residual,seconds=time.monotonic()-t));out['seconds']=time.monotonic()-begin;save()
 out['complete']=True;out['seconds']=time.monotonic()-begin;verify();save();print(ch,order,startindex,'COMPLETE',out['seconds'],flush=True)
except Exception as err:
 out['error']=dict(type=type(err).__name__,message=str(err));out['seconds']=time.monotonic()-begin;save();raise
