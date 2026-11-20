"""Fixed-state joint moment convenience; actual cancellation fallback."""
from pathlib import Path
import json,math,hashlib,importlib.util,sys
from numpy.polynomial.chebyshev import chebval
P=Path(__file__).resolve().parent;MU=.17446333200826397
# Filled once from the prevalidation lock, never inferred from mutable model data.
MODEL_SHA='f881ee6f9733fef2120fe5c75cb38f7a8e44bd85f88bba4d3d80a829c690a6f0'
REFERENCE_SHA='66c2e1335dcde2ca0adc1f7519abc8ab6748bcfc324e149e66c4facfed860fce'
def model():
 raw=(P/'model.json').read_bytes()
 if hashlib.sha256(raw).hexdigest()!=MODEL_SHA:raise RuntimeError('Frozen model changed')
 m=json.loads(raw)
 for p,s in m['source_pins'].items():
  if hashlib.sha256((P.parent/p).read_bytes()).hexdigest()!=s:raise RuntimeError('Frozen physical source changed')
 return m

def evaluate(E,*,T=2.,mu_e=MU,flavor='nu_e',targets=('e-','e+'),outgoing='unblocked',method='auto'):
 if type(E)not in(int,float)or not math.isfinite(E)or not 1<=E<=30 or type(T)not in(int,float)or type(mu_e)not in(int,float)or T!=2. or mu_e!=MU or flavor!='nu_e'or targets!=('e-','e+')or outgoing!='unblocked'or method not in['auto','detailed','diagnostic_fit']:raise ValueError('Only fixed physical state/domain/closure supported')
 m=model();x=2*math.log(E)/math.log(30)-1
 R=math.exp(chebval(x,m['models']['rate']['selected']['coefficients']));A=math.exp(chebval(x,m['models']['transport_rate']['selected']['coefficients']));g=float(chebval(x,m['models']['g']['selected']['coefficients']));Q=R*(E+2*g)
 bad=not all(math.isfinite(v)and v>0 for v in[R,A,Q])or A>2*R;near=abs(g)<=.08
 fallback=method=='detailed'or(method=='auto'and(near or bad))
 if fallback:
  if hashlib.sha256((P/'reference.py').read_bytes()).hexdigest()!=REFERENCE_SHA:raise RuntimeError('Actual fallback adapter changed')
  sp=importlib.util.spec_from_file_location('joint_actual_fallback_reference',P/'reference.py');ref=importlib.util.module_from_spec(sp);sys.modules[sp.name]=ref;sp.loader.exec_module(ref);row=ref.detailed(E)
  return dict(E=E,**row['total'],g=row['total']['neutrino_energy_change']/(2*row['total']['rate']),method='actual detailed CM quadrature',fallback=True,fallback_reason='explicit'if method=='detailed'else'near cancellation or physical-bound guard',controls=row['controls'],source_pins=row['pins'],seconds=row['seconds'],state=m['state'],units=m['units'],model_sha=MODEL_SHA,qualification=m['qualification'])
 if bad:raise ArithmeticError('Fitted physical bounds failed')
 return dict(E=E,rate=R,transport_rate=A,outgoing_energy_moment=Q,neutrino_energy_change=2*R*g,g=g,method='joint Chebyshev diagnostic'if method=='diagnostic_fit'else'joint Chebyshev moments',fallback=False,near_cancellation_guard=near,state=m['state'],units=m['units'],model_sha=MODEL_SHA,qualification=m['qualification'])
