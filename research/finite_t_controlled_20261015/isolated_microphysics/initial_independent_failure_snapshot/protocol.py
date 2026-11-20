"""Strict stdlib-only wire contract for the selected research kernels."""
from __future__ import annotations
import json,math,hashlib,sys,platform,importlib.metadata
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
CONVENTIONS=dict(material='physical-mass free particle-only n,p plus equilibrium e-/e+ pairs; muons/photons absent',rest='all material energies and chemical potentials include rest mass; no added Q',radiation='empty incoming antinu_e, freely escaping; ONLY neutron beta + positron capture',current='accepted physical-mass full one-body V/A/F2/GP current; no radiative/Coulomb/NN corrections',weak_parameters=dict(GF_MeV_minus2=1.1663785e-11,Vud=.974,gV=1.,gA=1.2723,xi=3.706,bare_weak_mass_MeV=(939.56542052+938.27208816)/2,vector_mass_MeV=840.,axial_mass_MeV=1000.,pion_mass_MeV=139.57),units='T,mu MeV; n fm^-3; epsilon,P MeV fm^-3; entropy kB fm^-3; R fm^-3 s^-1; Q MeV fm^-3 s^-1')
CONTROLS={'base':dict(pc=[28,44,10,60.],beta=[16,12,6,8],eos=dict(rtol=2e-12,tail=60.,anti_max_ratio=1e-8)),'fine':dict(pc=[32,52,12,70.],beta=[24,20,10,12],eos=dict(rtol=2e-12,tail=60.,anti_max_ratio=1e-8))}
def numeric(v):return type(v)in(int,float)and math.isfinite(v)
def exactkeys(v,keys):
 if not isinstance(v,dict)or set(v)!=set(keys):raise ValueError('Unexpected or missing schema fields')
def loads(s):
 def pairs(items):
  d={}
  for k,v in items:
   if k in d:raise ValueError('Duplicate JSON key')
   d[k]=v
  return d
 def bad(x):raise ValueError('Nonfinite JSON constant')
 return json.loads(s,object_pairs_hook=pairs,parse_constant=bad)
def dumps(x):return json.dumps(x,allow_nan=False,sort_keys=True,separators=(',',':'))
def digest(x):return hashlib.sha256(dumps(x).encode()).hexdigest()
def runtime():return dict(python=sys.version,numpy=importlib.metadata.version('numpy'),scipy=importlib.metadata.version('scipy'),platform=platform.platform())
def verify_pins(pins):
 exactkeys(pins,['sources','runtime'])
 if pins['runtime']!=runtime():raise ValueError('Runtime identity mismatch')
 for name,sha in pins['sources'].items():
  path=(ROOT/name).resolve()
  if not path.is_relative_to(ROOT)or path.suffix!='.py'or not path.is_file()or hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('Source identity mismatch: '+name)
def validate_request(r,pins):
 exactkeys(r,['version','state','controls','conventions','identity'])
 if r['version']!='nmb-isolated-1'or r['identity']!=digest(pins)or r['conventions']!=CONVENTIONS or r['controls']not in list(CONTROLS.values()):raise ValueError('Unsupported identity/convention/controls')
 exactkeys(r['state'],['T','nB','Ye']);T,n,Y=[r['state'][k]for k in ['T','nB','Ye']]
 if not all(numeric(x)for x in(T,n,Y))or not 1.9<=T<=2.1 or n!=1e-7 or not .28<=Y<=.34:raise ValueError('Only the selected local research bath is exposed')
def represented(log,*,zero_reason=None):
 if zero_reason is not None:return dict(status='zero',value=0.,log=None,reason=zero_reason)
 if not numeric(log):raise ValueError('Finite positive-moment logarithm required')
 try:value=math.exp(log)
 except OverflowError:value=None
 if value is None:return dict(status='unrepresented',value=None,log=log,reason='overflow')
 if value==0:return dict(status='unrepresented',value=None,log=log,reason='underflow')
 return dict(status='positive',value=value,log=log,reason=None)
def validate_positive(v):
 exactkeys(v,['status','value','log','reason'])
 if v['status']=='positive':
  if not numeric(v['value'])or v['value']<=0 or not numeric(v['log'])or v['reason']is not None or abs(math.log(v['value'])-v['log'])>2e-12:raise ValueError('Invalid positive/log pair')
 elif v['status']=='zero':
  if v['value']!=0 or v['log']is not None or v['reason']!='empty_incident_radiation':raise ValueError('Unjustified exact zero')
 elif v['status']=='unrepresented':
  if v['value']is not None or not numeric(v['log'])or v['reason']not in('underflow','overflow'):raise ValueError('Invalid unrepresented logarithm')
  expect=represented(v['log'])
  if expect!=v:raise ValueError('False underflow/overflow claim')
 else:raise ValueError('Unknown numerical status')
