"""Weak-Euler plus exact-FD adiabatic density split, with independent work."""
from pathlib import Path
from dataclasses import asdict
import sys,math,json,hashlib,platform,time
import numpy as np
import scipy
from numpy.polynomial.legendre import leggauss
ROOT=Path(__file__).resolve().parents[1];P=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'fd_prescribed_density'));import density_element as density_eos
import element
import qualification_contract
SOURCES=('nr_baryon_eos/free_nucleons.py','thermodynamics/lepton_reference.py','response_reference/free_density_response.py','allowed_opacity/allowed_capture.py','weak_rates/static_capture.py','capture_endpoint/mapped_capture.py','crossed_opacity/crossed_decay.py','quantum_slab/fixed_bath.py','quantum_slab/guarded_capture.py','fd_fluid_element/element.py','fd_prescribed_density/density_element.py','fd_fluid_element/qualification_contract.py','fd_fluid_element/qualify_source.py','fd_fluid_element/short_evolution.py','fd_density_feedback/engine.py','fd_density_feedback/run_feedback.py')
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def physical_provenance():return dict(source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in SOURCES},environment=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()))

def pressure_work(T,n0,Y,n1,*,order=4):
 before=density_eos.state(T,n0,Y);target=before['entropy_per_baryon'];extent=math.log(n1/n0);xs,ws=leggauss(order);value=0.
 for x,w in zip(xs,ws):
  n=n0*math.exp(.5*extent*(x+1));temp=density_eos.temperature_at_entropy(target,n,Y,T)
  value+=.5*extent*w*density_eos.state(temp,n,Y)['P_MeV_fm3']/n
 return float(value)

def weak_entropy_integral(before,u1,Y1,*,order=4):
 """Independent first-law integral on the actual straight u,Ye weak-kick path."""
 T0,n,Y0=before['T'],before['nB'],before['Ye'];u0=before['u_MeV_per_baryon'];du=u1-u0;dY=Y1-Y0;xs,ws=leggauss(order);value=0.
 for x,w in zip(xs,ws):
  v=.5*(x+1);Y=Y0+v*dY;u=u0+v*du;T=element.recover_temperature(u,n,Y,T0);s=density_eos.state(T,n,Y);m=s['matter'];affinity=m.proton.mu+m.electron.chemical_potential-m.neutron.mu
  value+=.5*w*(du-affinity*dY)/T
 return float(value)

def source_record(T,n,Y,provenance,initial=None,outer=6):
 if physical_provenance()!=provenance:raise ValueError('Physical source/integrator/environment changed during run')
 s=density_eos.state(T,n,Y)
 request=dict(T=T,nB=n,Ye=Y,full_fd_state=asdict(s['matter']),photons=False,outer_order=outer,internal_order=48,tail=40.,edges=list(element.EDGES),provenance=provenance)
 if initial is not None:
  if not (T==2. and n==1e-4 and Y==.3 and outer==6):raise ValueError('Initial qualification cannot be reused at changed state/controls')
  q=qualification_contract.load(ROOT/'fd_fluid_element/source_qualification.json',qualification_contract.fingerprint(T,n,Y))
  result=q['coarse_source'];origin='full fingerprint verified initial physical source'
 else:result=element.source(T,n,Y,outer_order=outer);origin='fresh full physical spectral quadrature'
 for k,v in [('T',T),('nB',n),('Ye',Y),('outer_order',outer),('internal_order',48),('tail',40.),('edges',list(element.EDGES))]:
  if result[k]!=v:raise ValueError('Physical source state/control mismatch')
 return dict(request=request,fingerprint=digest(request),result=result,origin=origin)
