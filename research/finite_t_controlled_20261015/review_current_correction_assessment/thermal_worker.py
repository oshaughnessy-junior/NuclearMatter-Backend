"""One variant/profile, independent EOS, amplitude injection only in child."""
from pathlib import Path
import sys,json,importlib.util,hashlib,time,os
import numpy as np
from independent_eos import eos
P=Path(__file__).resolve().parent;R=P.parent;A=R/'current_correction_assessment';sys.dont_write_bytecode=True
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
mode=sys.argv[1];fine=int(sys.argv[2]);assert mode in ['full','F2zero']and fine in [0,1]
lock=json.loads((A/'LOCK.json').read_text());pins=lock['pins']
for rel,h in pins.items():assert sha(R/rel)==h
EC=load('isolated_review_current_ec',R/'hot_electron_capture_moments/electron_capture.py');PC=load('isolated_review_current_pc',R/'review_pc_moments/reference.py');K=load('isolated_review_owned_currents',A/'configured_currents.py')
original_ec=EC.scalar_amplitude;original_pc=PC.amplitude
# This child never imports the author worker or any orchestrator/driver.
if mode=='F2zero':EC.scalar_amplitude=K.ec_F2zero;PC.amplitude=K.pc_F2zero
T,n,y=3.,1e-7,.5490065797957709;b,s=eos(T,n,y);accepted,_=EC.matched(T,n,y)
start=time.monotonic();ec=EC.integrate(EC.Bath(**b),proton_order=30 if fine else 24,electron_order=50 if fine else 40,initial_order=12 if fine else 10,outgoing_order=12 if fine else 10,phi_order=12 if fine else 10,tail=70. if fine else 55.,scenarios=[dict(name='empty',mode='empty')]);pc=PC.integral(b,order_n=34 if fine else 30,order_e=54 if fine else 48,order_angle=12 if fine else 10,tail=70. if fine else 55.)
channels=dict(EC=ec['scenarios']['empty']['net'],PC=dict(zip(PC.NAMES,map(float,np.asarray(pc['sums'])[0,0]))))
for rel,h in pins.items():assert sha(R/rel)==h
out=dict(mode=mode,fine=fine,bath=b,eos=s,accepted_bath=vars(accepted),channels=channels,density=dict(EC=ec['proton_density']/(n*y)-1,PC=pc['neutron_density']/(n*(1-y))-1),pid=os.getpid(),parent_pid=os.getppid(),seconds=time.monotonic()-start,raw=dict(EC=ec,PC=pc),child_only_override=mode=='F2zero',original_function_modules=[original_ec.__module__,original_pc.__module__],configured_sha256=sha(A/'configured_currents.py'))
path=P/f'thermal_{mode}_{fine}.json';assert not path.exists();path.write_text(json.dumps(out,indent=2)+'\n');print(mode,fine,out['seconds'])
