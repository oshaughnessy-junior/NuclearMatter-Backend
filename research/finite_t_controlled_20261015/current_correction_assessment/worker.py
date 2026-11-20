"""One current variant/profile in an isolated process. No upstream writes."""
from pathlib import Path
import sys,json,hashlib,importlib.util,time
P=Path(__file__).resolve().parent;ROOT=P.parent;sys.dont_write_bytecode=True
lock=json.loads((P/'LOCK.json').read_text())
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m

def run(variant,profile):
 if variant not in['baseline','F2zero']or profile not in[0,1]:raise ValueError('Locked current and profile only')
 path=P/f'{variant}_profile{profile}.json'
 if path.exists():raise RuntimeError('Immutable source output must not be overwritten')
 for p,h in lock['pins'].items():
  if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:raise RuntimeError('Pinned input changed '+p)
 C=load('isolated_F2_EC_phase',ROOT/'hot_electron_capture_moments/electron_capture.py');PC=load('isolated_F2_PC_phase',ROOT/'review_pc_moments/reference.py');K=load('isolated_owned_configured_current',P/'configured_currents.py');original_ec=C.scalar_amplitude;original_pc=PC.amplitude
 # Deliberate explicit injection exists ONLY within this fresh child process.
 C.scalar_amplitude=K.ec_full if variant=='baseline'else K.ec_F2zero;PC.amplitude=K.pc_full if variant=='baseline'else K.pc_F2zero
 start=time.monotonic();b,st=C.matched(**lock['state']);c=lock['profiles'][profile];po,eo,ia,oa,ph,tail=c['EC'];nr,er,ar,pt=c['PC'];ec=C.integrate(b,proton_order=po,electron_order=eo,initial_order=ia,outgoing_order=oa,phi_order=ph,tail=tail,scenarios=[dict(name='empty',mode='empty')]);pc=PC.integral(vars(b),order_n=nr,order_e=er,order_angle=ar,tail=pt)
 values=dict(EC=[ec['scenarios']['empty']['net']['number'],ec['scenarios']['empty']['net']['neutrino_energy']],PC=pc['sums'][0][0][:2]);n=lock['state']['nB'];y=lock['state']['Ye'];out=dict(variant=variant,profile=profile,state=lock['state'],bath=vars(b),EOS=st,values=values,raw=dict(EC=ec,PC=pc),density_residuals=dict(EC=ec['proton_density']/(n*y)-1,PC=pc['neutron_density']/(n*(1-y))-1),configured_current_sha256=hashlib.sha256((P/'configured_currents.py').read_bytes()).hexdigest(),worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),phase_sourcepins=lock['pins'],seconds=time.monotonic()-start,override='Explicit owned amplitude injection in isolated child; original physical phase/current source file bytes are unchanged. Raw phase amplitude label scalar is qualified by THIS variant/provenance.',units='number fm^-3s^-1; neutrino energy MeVfm^-3s^-1')
 for p,h in lock['pins'].items():
  if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:raise RuntimeError('Pinned input changed after '+p)
 path.write_text(json.dumps(out,indent=2)+'\n');print(variant,profile,values,out['seconds'])
if __name__=='__main__':run(sys.argv[1],int(sys.argv[2]))
