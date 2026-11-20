"""Full-state/source/control provenance for a reused physical qualification."""
from __future__ import annotations
from pathlib import Path
from dataclasses import asdict
import hashlib,json,sys,platform
import numpy as np
import scipy
import element
P=Path(__file__).resolve().parent
UPSTREAM=('quantum_slab/fixed_bath.py','quantum_slab/guarded_capture.py','capture_endpoint/mapped_capture.py','crossed_opacity/crossed_decay.py','allowed_opacity/allowed_capture.py','response_reference/free_density_response.py','nr_baryon_eos/free_nucleons.py','thermodynamics/lepton_reference.py','weak_rates/static_capture.py')
LOCAL=('element.py','qualify_source.py','short_evolution.py','qualification_contract.py')
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def fingerprint(T,nB,Ye,*,coarse_order=6,fine_order=10,internal_order=48,tail=40.,edges=element.EDGES):
 s=element.eos(T,nB,Ye,photons=False)
 hashes={f:hashlib.sha256((P.parent/f).read_bytes()).hexdigest() for f in UPSTREAM}
 hashes.update({f:hashlib.sha256((P/f).read_bytes()).hexdigest() for f in LOCAL})
 value=dict(schema=1,input_state=dict(T=T,nB=nB,Ye=Ye,photons=False),full_fd_state=asdict(s['matter']),thermodynamic_closure={k:s[k] for k in ('u_MeV_per_baryon','u_T','u_Y')},source_sha256=hashes,environment=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()),controls=dict(coarse_order=coarse_order,fine_order=fine_order,internal_order=internal_order,tail=tail,edges=list(edges),inventory=['electron_capture','positron_capture','crossed_neutron_decay'],radiation='empty_one_helicity_each_species'))
 digest(value);return value

def stamp(report,expected):
 return report|dict(qualification_schema=1,qualification_fingerprint=expected,qualification_fingerprint_sha256=digest(expected),qualified_payload_sha256=digest(report))

def validate(report,expected):
 if report.get('qualification_schema')!=1:raise ValueError('Missing full qualification provenance; legacy payload excluded')
 if report.get('qualification_fingerprint')!=expected or report.get('qualification_fingerprint_sha256')!=digest(expected):raise ValueError('Stale qualification state/source/environment/controls')
 payload={k:v for k,v in report.items() if k not in ('qualification_schema','qualification_fingerprint','qualification_fingerprint_sha256','qualified_payload_sha256')}
 if digest(payload)!=report.get('qualified_payload_sha256'):raise ValueError('Qualification payload digest mismatch')
 if payload.get('passed') is not True:raise ValueError('Physical qualification did not pass')
 state=expected['input_state'];controls=expected['controls']
 for name,order in [('coarse_source',controls['coarse_order']),('fine_source',controls['fine_order'])]:
  source=payload[name]
  want=dict(T=state['T'],nB=state['nB'],Ye=state['Ye'],outer_order=order,internal_order=controls['internal_order'],tail=controls['tail'],edges=controls['edges'])
  if any(source.get(k)!=v for k,v in want.items()):raise ValueError('Qualified source metadata mismatch')
 return payload

def load(path,expected):
 if not Path(path).is_file():raise ValueError('Missing physical qualification payload')
 return validate(json.loads(Path(path).read_text()),expected)
