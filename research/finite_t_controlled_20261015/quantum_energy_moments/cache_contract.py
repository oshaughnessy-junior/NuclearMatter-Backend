"""Exact scientific cache provenance; never infer compatibility from a tag."""
from __future__ import annotations
from pathlib import Path
from dataclasses import asdict
import hashlib,json,platform,sys
import numpy as np
import scipy
import integrated

P=Path(__file__).resolve().parent
UPSTREAM=('quantum_slab/fixed_bath.py','quantum_slab/guarded_capture.py','capture_endpoint/mapped_capture.py','crossed_opacity/crossed_decay.py','allowed_opacity/allowed_capture.py','response_reference/free_density_response.py','nr_baryon_eos/free_nucleons.py','thermodynamics/lepton_reference.py','weak_rates/static_capture.py')
LOCAL=('integrated.py','cache_contract.py','run_study.py')

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def fingerprint(matter,flavor,*,outer_order,internal_order=48,tail=40.,length_cm=1e5,edges=integrated.EDGES):
    integrated.validate(length_cm,outer_order,internal_order,tail,edges)
    if flavor not in ('nue','antinue'):raise ValueError('Expected nue/antinue')
    sources={s:hashlib.sha256((P.parent/s).read_bytes()).hexdigest() for s in UPSTREAM}
    sources.update({s:hashlib.sha256((P/s).read_bytes()).hexdigest() for s in LOCAL})
    result=dict(schema=1,state=asdict(matter),state_inputs=dict(T=matter.neutron.T,nn=matter.neutron.density,np=matter.proton.density,mass=matter.neutron.mass,offset_neutron=matter.neutron.offset,offset_proton=matter.proton.offset),
                source_sha256=sources,environment=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()),
                controls=dict(flavor=flavor,outer_order=outer_order,internal_order=internal_order,internal_tail=tail,length_cm=length_cm,edges=list(map(float,edges))))
    digest(result)  # Reject nonfinite state fields before any cache access.
    return result


def record(request,result):
    controls=request['controls']
    for a,b in [('flavor','flavor'),('outer_order','outer_order'),('internal_order','internal_order'),('internal_tail','internal_tail'),('length_cm','length_cm')]:
        if result.get(a)!=controls[b]:raise ValueError('Result controls disagree with cache fingerprint')
    edges=controls['edges'];panels=result.get('panels',[])
    if len(panels)!=len(edges)-1 or any(p['lo']!=lo or p['hi']!=hi for p,lo,hi in zip(panels,edges[:-1],edges[1:])):raise ValueError('Result energy panels disagree with fingerprint')
    return dict(fingerprint=request,fingerprint_sha256=digest(request),result=result,result_sha256=digest(result))


def load(path,expected):
    """Reject a missing/stale/malformed file in full before returning any hit."""
    path=Path(path)
    if not path.is_file():raise ValueError('Requested cache does not exist')
    data=json.loads(path.read_text())
    if data.get('cache_schema')!=1 or not isinstance(data.get('records'),list):raise ValueError('Missing exact cache provenance; legacy caches are excluded')
    hits={}
    for row in data['records']:
        request=row.get('fingerprint');key=row.get('fingerprint_sha256')
        if key not in expected or request!=expected[key] or digest(request)!=key:raise ValueError('Stale or unknown cache fingerprint')
        if key in hits:raise ValueError('Duplicate cache record')
        if row.get('result_sha256')!=digest(row.get('result')):raise ValueError('Cache result digest mismatch')
        record(request,row['result'])
        hits[key]=row['result']
    return hits


def save(path,records,elapsed_seconds):
    data=dict(cache_schema=1,records=records,elapsed_seconds=elapsed_seconds)
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
