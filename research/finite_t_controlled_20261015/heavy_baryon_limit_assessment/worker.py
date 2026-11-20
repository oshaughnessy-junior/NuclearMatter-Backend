"""Isolated consistent artificial baryon-mass family, one current/profile."""
from pathlib import Path
import sys,json,math,hashlib,importlib.util,time
P=Path(__file__).resolve().parent;ROOT=P.parent;sys.dont_write_bytecode=True;LOCK=json.loads((P/'LOCK.json').read_text())
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m

def run(scale,profile,method):
 if scale not in LOCK['scales']or profile not in[0,1]or method not in['scalar','spinor']or(method=='spinor'and(scale!=1 or profile!=0)):raise ValueError('Locked artificial family/profile/anchor only')
 path=P/f'lambda{int(scale)}_{method}_profile{profile}.json'
 if path.exists():raise RuntimeError('Immutable completed child')
 for p,h in LOCK['pins'].items():
  if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:raise RuntimeError('Pinned input changed '+p)
 C=load('heavy_mass_isolated_EC_phase',ROOT/'hot_electron_capture_moments/electron_capture.py');PC=load('heavy_mass_isolated_PC_phase',ROOT/'review_pc_moments/reference.py');K=load('heavy_mass_owned_point_current',P/'point_current.py');mn0,mp0=C.MN,C.MP;Q0=mn0-mp0;M0=(mn0+mp0)/2;M=scale*M0;mn=M+Q0/2;mp=M-Q0/2
 # Child-only complete phase configuration. C.matched passes THESE masses
 # explicitly to the unchanged free baryon density solver, including restoffset.
 C.MN=mn;C.MP=mp;C.MW=M;PC.MN=mn;PC.MP=mp;PC.MW=M
 C.scalar_amplitude=lambda k,n,l,p:K.amplitude(k,n,l,p,mn,mp,False)
 C.E.vertices=lambda q,weakM:K.vertices(q,weakM,C.E)
 if method=='scalar':PC.amplitude=lambda k,p,l,n:K.amplitude(k,p,l,n,mp,mn,True)
 else:
  eb=C.E.Bath(LOCK['state']['T'],mp,mn,0.,0.,0.,anti=True,ml=C.ME,Mweak=M)
  PC.amplitude=lambda k,p,l,n:C.E.amplitude(eb,k,p,l,n)/2
 start=time.monotonic();b,st=C.matched(**LOCK['state']);c=LOCK['profiles'][profile];po,eo,ia,oa,phi,tail=c['EC'];nr,er,ang,pt=c['PC'];ec=C.integrate(b,proton_order=po,electron_order=eo,initial_order=ia,outgoing_order=oa,phi_order=phi,tail=tail,amplitude=method,scenarios=[dict(name='empty',mode='empty')]);pc=PC.integral(vars(b),order_n=nr,order_e=er,order_angle=ang,tail=pt,budget=110.)
 vals=dict(EC=[ec['scenarios']['empty']['net']['number'],ec['scenarios']['empty']['net']['neutrino_energy']],PC=pc['sums'][0][0][:2]);T=LOCK['state']['T'];n=LOCK['state']['nB'];y=LOCK['state']['Ye'];out=dict(scale=scale,profile=profile,method=method,masses=dict(mn0=mn0,mp0=mp0,Q0=Q0,mn=mn,mp=mp,Mweak=M,actual_Q=mn-mp,lepton=C.ME),bath=vars(b),EOS=st,values=vals,raw=dict(EC=ec,PC=pc),density_residuals=dict(EC=ec['proton_density']/(n*y)-1,PC=pc['neutron_density']/(n*(1-y))-1),point_source_sha256=hashlib.sha256((P/'point_current.py').read_bytes()).hexdigest(),worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),seconds=time.monotonic()-start,configuration='Actual phase MN,MP,MW in BOTH modules plus actual FD roots and spinprojectors/amplitude masses deformed in CHILD ONLY. Raw original source pin labels need THIS override metadata; no upstream modification.',units='number fm^-3s^-1; neutrino energy MeVfm^-3s^-1')
 for p,h in LOCK['pins'].items():
  if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:raise RuntimeError('Pinned input changed after '+p)
 path.write_text(json.dumps(out,indent=2)+'\n');print(scale,method,profile,vals,out['seconds'])
if __name__=='__main__':run(float(sys.argv[1]),int(sys.argv[2]),sys.argv[3])
