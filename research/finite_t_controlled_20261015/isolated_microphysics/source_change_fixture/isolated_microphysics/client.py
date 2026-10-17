"""Stdlib-only isolated research client; no physical imports or source cache."""
from __future__ import annotations
from dataclasses import dataclass,asdict
import subprocess,sys,math
import importlib.util
from pathlib import Path
_spec=importlib.util.spec_from_file_location('isolated_microphysics_protocol',Path(__file__).with_name('protocol.py'))
P=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(P)
@dataclass(frozen=True)
class MaterialState:
 T:float
 nB:float
 Ye:float
@dataclass(frozen=True)
class Controls:
 profile:str='base'
 def asdict(self):
  if self.profile not in P.CONTROLS:raise ValueError('Unknown reviewed control profile')
  # Fresh structured copy, never a shared mutable public object.
  return P.loads(P.dumps(P.CONTROLS[self.profile]))
class Client:
 def __init__(self):self._pins=P.loads((P.HERE/'PINS.json').read_text());P.verify_pins(self._pins)
 def request(self,state,controls=Controls()):
  if type(state)is not MaterialState or type(controls)is not Controls:raise TypeError('Explicit immutable state and controls required')
  req=dict(version='nmb-isolated-1',state=asdict(state),controls=controls.asdict(),conventions=P.loads(P.dumps(P.CONVENTIONS)),identity=P.digest(self._pins));P.validate_request(req,self._pins);return req
 def validate(self,out,req):
  P.verify_pins(self._pins)
  P.validate_request(req,self._pins)
  P.exactkeys(out,['version','status','request','identity','runtime','eos','sources'])
  if out['version']!=req['version']or out['status']!='ok'or out['request']!=req or out['identity']!=P.digest(self._pins)or out['runtime']!=self._pins['runtime']:raise ValueError('Response provenance/state/control mismatch')
  e=out['eos'];P.exactkeys(e,['energy_density','pressure','entropy_density','cv','densities','total_mu','masses','density_residual'])
  for k in ['energy_density','pressure','entropy_density','cv','density_residual']:
   if not P.numeric(e[k])or e[k]<0:raise ValueError('Invalid EOS scalar')
  if e['cv']<=0 or e['density_residual']>2e-9 or e['masses']!=dict(neutron=939.56542052,proton=938.27208816,electron=.51099895069):raise ValueError('EOS physical forward mismatch')
  d=e['densities'];P.exactkeys(d,['neutron','proton','electron','positron']);P.exactkeys(e['total_mu'],['neutron','proton','electron'])
  if not all(P.numeric(v)for v in e['total_mu'].values())or not all(P.numeric(v)and v>0 for v in d.values()):raise ValueError('Nonfinite density/chemical potential')
  s=req['state'];res=max(abs(d['neutron']/(s['nB']*(1-s['Ye']))-1),abs(d['proton']/(s['nB']*s['Ye'])-1),abs((d['electron']-d['positron'])/(s['nB']*s['Ye'])-1))
  if res>2e-9:raise ValueError('Reconstructed density/charge mismatch')
  z=out['sources'];P.exactkeys(z,['channels','total','inverse','signed','residuals']);P.exactkeys(z['channels'],['positron_capture','neutron_beta'])
  for branch in [*z['channels'].values(),z['total']]:
   P.exactkeys(branch,['number','energy'])
   for v in branch.values():
    P.validate_positive(v)
    if v['status']!='positive':raise ValueError('Each admitted emission channel must be represented positive')
  P.validate_positive(z['inverse'])
  if z['inverse']['status']!='zero':raise ValueError('Empty-radiation inverse must be exactly zero')
  for key in ['number','energy']:
   vals=[branch[key]['value']for branch in z['channels'].values()];total=z['total'][key]['value']
   if total is None or any(v is None for v in vals)or abs(sum(vals)/total-1)>2e-12:raise ValueError('Selected source channels do not sum')
  R=z['total']['number']['value'];Q=z['total']['energy']['value'];expected=dict(neutron=-R,proton=R,electron_net=R,antineutrino_e=R,material_energy=-Q,radiation_energy=Q,material_Le=R,radiation_Le=-R)
  P.exactkeys(z['signed'],expected)
  if not all(P.numeric(z['signed'][k])and abs(z['signed'][k]/v-1)<=2e-12 for k,v in expected.items()):raise ValueError('Signed material/radiation ledger mismatch')
  P.exactkeys(z['residuals'],['pc_density','beta_density','energy'])
  if not all(P.numeric(v)and abs(v)<=3e-8 for v in z['residuals'].values()):raise ValueError('Source forward residual failed')
  return out
 def evaluate(self,state,controls=Controls()):
  req=self.request(state,controls);P.verify_pins(self._pins)
  done=subprocess.run([sys.executable,'-I','-B',str(P.HERE/'worker_entry.py')],input=P.dumps(req),text=True,capture_output=True,timeout=110,cwd=str(P.HERE))
  if done.returncode!=0 or done.stderr or len(done.stdout)>2000000:raise RuntimeError('Isolated worker failed: '+done.stdout[:2000]+done.stderr[:1000])
  return self.validate(P.loads(done.stdout),req)
