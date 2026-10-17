"""Fresh isolated physical sources supplied with the newly bound material's total mu."""
from pathlib import Path
import sys,json,importlib.util,hashlib,math,time
P=Path(__file__).resolve().parent;R=P.parent;sys.dont_write_bytecode=True;sys.path.insert(0,str(P));import matter as M
assert json.loads((P/'forward_results.json').read_text())['status']=='PASS';assert json.loads((P/'partial_results.json').read_text())['status']=='PASS'
sp=importlib.util.spec_from_file_location('binding_frozen_three_source_functions',R/'three_source_cooling_element/worker.py');W=importlib.util.module_from_spec(sp);sys.modules[sp.name]=W;sp.loader.exec_module(W);pins=W.pins();B=W.physical()
q=M.LOCK['weak_point'];s=M.state(q['T'],q['nB'],q['Ye'],photons=q['photons']);profile=sys.argv[1]
if profile not in['base','fine']:raise ValueError('Locked source profiles only')
# This helper calls only physical event integrals, not its legacy state() factory.
inp=dict(T=s.T,nB=s.nB,Ye=s.Ye,bath=s.weak_bath(),u_Y=s.e_Y/s.nB,u_T=s.cv/s.nB)
start=time.monotonic();z=W.source(B,inp,profile);Rnet=z['R'];Q=z['Q'];ndot=q['ndot'];Ydot=Rnet/s.nB;Tdot=(((s.energy+s.pressure)/s.nB-s.e_n)*ndot-Q-s.e_Y*Ydot)/s.cv
# Rest-inclusive internal species field energy changes use the exact event output.
edot=s.cv*Tdot+s.e_n*ndot+s.e_Y*Ydot;firstlaw=(s.energy+s.pressure)*ndot/s.nB-Q;scale=abs(Q)+abs((s.energy+s.pressure)*ndot/s.nB)
checks=[]
def ck(name,error,gate):checks.append(dict(name=name,error=float(error),gate=gate,passed=bool(abs(error)<=gate)))
ck('instantaneous first law reconstruction',(edot-firstlaw)/scale,2e-9);ck('weak energy species',z['species_energy_ledger'],2e-9);ck('weak direct energy',z['direct_energy_ledger'],2e-9);nn=z['number_sources'];gross=sum(z['rates'].values());ck('electron family Le',(nn['electron_net']+nn['nue']-nn['antinue'])/gross,2e-9);ck('baryon count',(nn['n']+nn['p'])/gross,2e-9);ck('charge',(nn['p']-nn['electron_net'])/gross,2e-9)
# Actual pairs are rethermalized by internal EM equilibrium. Equal pair source
# is independently recovered from particle and positron thermodynamic derivatives.
h=2e-4;sa=M.state(s.T*(1+h),s.nB,s.Ye,photons=s.photons);sb=M.state(s.T*(1-h),s.nB,s.Ye,photons=s.photons);sn1=M.state(s.T,s.nB*(1+h),s.Ye,photons=s.photons);sn0=M.state(s.T,s.nB*(1-h),s.Ye,photons=s.photons);sy1=M.state(s.T,s.nB,s.Ye+h,photons=s.photons);sy0=M.state(s.T,s.nB,s.Ye-h,photons=s.photons)
def rate(field):return (getattr(sa.electron,field)-getattr(sb.electron,field))/(2*h*s.T)*Tdot+(getattr(sn1.electron,field)-getattr(sn0.electron,field))/(2*h*s.nB)*ndot+(getattr(sy1.electron,field)-getattr(sy0.electron,field))/(2*h)*Ydot
# Subtract compression dilution and weak particle creation to isolate EM pairs.
emminus=rate('particle_density')-s.electron.particle_density*ndot/s.nB-nn['electron_particle'];emplus=rate('antiparticle_density')-s.electron.antiparticle_density*ndot/s.nB-nn['positron'];ck('EM pair source equal', (emminus-emplus)/max(gross,abs(emminus),abs(emplus)),3e-5)
no=M.state(s.T,s.nB,s.Ye,photons=False);Tdot_off=(((no.energy+no.pressure)/no.nB-no.e_n)*ndot-Q-no.e_Y*Ydot)/no.cv
loaded={}
for mod in list(sys.modules.values()):
 f=getattr(mod,'__file__',None)
 if f:
  p=Path(f).resolve()
  if p.suffix=='.py'and p.is_relative_to(R):loaded[str(p.relative_to(R))]=hashlib.sha256(p.read_bytes()).hexdigest()
W.pins();out=dict(status='PASS'if all(c['passed']for c in checks)else'FAIL',profile=profile,state=s.asdict(),source_input=inp,source=z,ndot=ndot,Ye_dot=Ydot,T_dot=Tdot,energy_dot=edot,firstlaw_energy_dot=firstlaw,photon_comparison=dict(Cv_on=s.cv,Cv_off=no.cv,Tdot_on=Tdot,Tdot_off=Tdot_off,ansatz='same instantaneous weak moments; mechanical density history, internal trapped photons vs absent'),EMpair_sources=dict(electron=emminus,positron=emplus),checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],loaded_source_hashes=loaded,accepted_source_pins=pins,seconds=time.monotonic()-start)
(P/f'weak_{profile}.json').write_text(json.dumps(out,indent=2)+'\n');print(out['status'],out['passed'],out['failed'],'seconds',out['seconds'])
