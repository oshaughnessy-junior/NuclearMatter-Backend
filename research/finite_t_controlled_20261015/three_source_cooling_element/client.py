"""Stdlib fresh-process client; exact selected source/state/control identity."""
from pathlib import Path
import subprocess,sys,os,json,math,hashlib,importlib.util
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
s=importlib.util.spec_from_file_location('three_source_protocol_stdlib',HERE/'worker.py');W=importlib.util.module_from_spec(s);sys.modules[s.name]=W;s.loader.exec_module(W)
def finite_tree(v):
 if isinstance(v,dict):return all(finite_tree(x)for x in v.values())
 if isinstance(v,list):return all(finite_tree(x)for x in v)
 return type(v)not in(int,float)or math.isfinite(v)
class Client:
 def __init__(self):self.pin=W.pins();self.identity=W.digest(self.pin);self.calls=[]
 def evaluate(self,T,Ye,*,action='state',profile='base',photons=True):
  q=dict(T=T,Ye=Ye,nB=W.NB,action=action,profile=profile,photons=photons,identity=self.identity);W.request(q,self.pin);W.pins();env=dict(os.environ);env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1');done=subprocess.run([sys.executable,'-I','-B',str(HERE/'worker.py')],input=json.dumps(q,allow_nan=False),text=True,capture_output=True,timeout=110,cwd=str(HERE),env=env)
  if done.returncode or done.stderr or len(done.stdout)>3_000_000:raise RuntimeError('Fresh isolated physics failed: '+done.stdout[:2000]+done.stderr[:1000])
  out=W.loads(done.stdout);self.validate(out,q);self.calls.append(dict(action=action,T=T,Ye=Ye,profile=profile,elapsed=out['elapsed']));return out
 def validate(self,o,q):
  W.pins()
  if set(o)!=set(['request','identity','state','loaded_sources','sources','elapsed'])or o['request']!=q or o['identity']!=self.identity or not finite_tree(o):raise ValueError('Response state/provenance/finite mismatch')
  if set(o['loaded_sources'])!=set(self.pin['expected_loaded_python']):raise ValueError('Incomplete loaded-source identity')
  for rel,h in o['loaded_sources'].items():
   if self.pin['files'].get(rel)!=h:raise ValueError('Response unpinned dependency')
  s=o['state']
  if(s['T'],s['Ye'],s['nB'],s['photons'])!=(q['T'],q['Ye'],q['nB'],q['photons'])or s['density_residual']>2e-9 or s['u_T']<=0 or s['u']<=0:raise ValueError('Actual EOS mismatch')
  raw=s['material'];nb=q['nB'];Y=q['Ye']
  if any(type(s[k])not in(int,float)or not math.isfinite(s[k])for k in['u','energy_density','cv','u_T','u_Y','pressure','entropy_density']):raise ValueError('Finite real EOS scalars required')
  eg=math.pi**2*q['T']**4/(15*197.3269804**3)if q['photons']else 0.
  ee=sum(raw[k]['energy_density']for k in['neutron','proton','electron'])+eg;cv=raw['neutron']['cv_fixed_density']+raw['proton']['cv_fixed_density']+raw['electron']['cv_fixed_net_density']+4*eg/q['T']
  if abs(s['energy_density']/ee-1)>2e-12 or abs(s['u']*nb/ee-1)>2e-12 or abs(s['cv']/cv-1)>2e-12 or abs(s['u_T']*nb/cv-1)>2e-12 or s['photon']['energy_density']!=eg:raise ValueError('Photon/material energy or Cv accounting mismatch')
  d=[raw['neutron']['density']/(nb*(1-Y))-1,raw['proton']['density']/(nb*Y)-1,(s['electron_particle_density']-s['positron_density'])/(nb*Y)-1]
  if max(abs(v)for v in d)>2e-9:raise ValueError('Reconstructed particle/charge mismatch')
  if q['action']=='state':
   if o['sources']is not None:raise ValueError('Unexpected source payload')
   return o
  z=o['sources'];channels=z['channels']
  if z['profile']!=q['profile']or z['controls']!=W.CONTROLS[q['profile']]or set(channels)!=set(['beta','PC','EC']):raise ValueError('Source control/channel mismatch')
  if any(type(v)not in(int,float)or not math.isfinite(v)or v<=0 for ch in channels.values()for v in ch.values()):raise ValueError('Positive separate channel moments required')
  R=channels['beta']['number']+channels['PC']['number']-channels['EC']['number'];Q=sum(v['neutrino_energy']for v in channels.values());gross=sum(v['number']for v in channels.values())
  if abs(z['R']-R)>2e-12*gross or abs(z['Q']/Q-1)>2e-12 or max(abs(v)for v in z['density_residuals'].values())>3e-8 or abs(z['direct_energy_ledger'])>2e-10 or abs(z['species_energy_ledger'])>2e-10:raise ValueError('Source forward/number/energy mismatch')
  expected_u=-Q/nb;expected_y=R/nb
  if any(type(z[k])not in(int,float)or not math.isfinite(z[k])for k in['u_dot','Ye_dot','T_dot'])or abs(z['u_dot']/expected_u-1)>2e-12 or abs(z['Ye_dot']-expected_y)>2e-12*gross/nb or abs(z['T_dot']-((expected_u-s['u_Y']*expected_y)/s['u_T']))>2e-12*max(1.,abs(z['T_dot'])):raise ValueError('Conservative source derivative/sign mismatch')
  n=z['number_sources']
  if max(abs(n['p']-R),abs(n['n']+R),abs(n['electron_net']-R),abs(n['nue']-channels['EC']['number']),abs(n['antinue']-channels['beta']['number']-channels['PC']['number']),abs(n['electron_particle']-channels['beta']['number']+channels['EC']['number']),abs(n['positron']+channels['PC']['number']))>2e-12*gross:raise ValueError('Species/lepton ledger mismatch')
  return o
