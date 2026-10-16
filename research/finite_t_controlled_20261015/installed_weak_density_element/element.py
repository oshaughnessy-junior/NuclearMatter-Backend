"""Short prescribed-density element: fresh full-current sources, conservative u/Ye."""
from __future__ import annotations
from pathlib import Path
import sys,json,math,time,hashlib,importlib.util
import numpy as np
from scipy.integrate import simpson
P=Path(__file__).resolve().parent;ROOT=P.parent;sys.dont_write_bytecode=True;L=json.loads((P/'LOCK.json').read_text());PIN=json.loads((P/'UPSTREAM.json').read_text())
def verify():
 for rel,h in PIN.items():
  if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=h:raise RuntimeError('Frozen dependency changed '+rel)
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
verify();M=load('weak_density_accepted_binding',ROOT/L['binding']);W=load('weak_density_frozen_source_functions',ROOT/'three_source_cooling_element/worker.py');W.pins();B=W.physical()
NAMES=['u','Ye','E_nue','E_antinue','N_nue','N_antinue','work','material_entropy_integral','weak_electron','weak_positron']
def density(t):return L['initial']['nB']*math.exp(L['density']['k']*t)
def invert(t,u,Y,Tguess):
 hist=[];T=float(Tguess)
 for _ in range(8):
  s=M.state(T,density(t),float(Y),photons=True);res=s.specific_energy-u;hist.append(dict(T=T,residual_u=res))
  if abs(res)<=L['gates']['EOS_inverse_absolute_u']:return s,hist
  T-=res/(s.cv/s.nB)
 raise ArithmeticError('Actual EOS energy inversion unresolved')
def sources(s,profile='base'):
 # This imports frozen function definitions, not a driver or legacy state factory.
 q=dict(T=s.T,nB=s.nB,Ye=s.Ye,bath=s.weak_bath(),u_Y=s.e_Y/s.nB,u_T=s.cv/s.nB)
 # Frozen source helper's NB is fixed1e-7 in its density diagnostics; the actual
 # density must vary here, so assemble the same eventcalls with explicit nB.
 ep=W.CONTROLS[profile]['EC'];pp=W.CONTROLS[profile]['PC'];bp=W.CONTROLS[profile]['beta'];b=s.weak_bath();n=s.nB
 ec=B.EC.integrate(B.EC.Bath(**b),proton_order=ep[0],electron_order=ep[1],initial_order=ep[2],outgoing_order=ep[2],phi_order=ep[2],tail=ep[3],scenarios=[dict(name='empty',mode='empty')],budget_seconds=90.)
 pc=B.PC.integral(b,order_n=pp[0],order_e=pp[1],order_angle=pp[2],tail=pp[3],budget=90.)
 beta=B.BETA.integrate(B.BETA.K.Bath(**b),rest_order=bp[0],parent_order=bp[1],orient_order=bp[2],phi_order=bp[3],coordinate=bp[4],parent_coordinate=bp[5],tail=bp[6],scenarios=[dict(name='empty',mode='empty')],budget_seconds=90.)
 ch=dict(EC=ec['scenarios']['empty']['net'],PC=dict(zip(B.PC.NAMES,map(float,pc['sums'][0][0]))),beta=beta['scenarios']['empty']['net']);rates={k:v['number']for k,v in ch.items()};energy={k:v['neutrino_energy']for k,v in ch.items()};R=rates['beta']+rates['PC']-rates['EC'];Q=sum(energy.values());loss=sum(v['material_energy_loss']for v in ch.values());en=ch['EC']['neutron_energy']-ch['PC']['neutron_energy']-ch['beta']['neutron_energy'];ep=ch['PC']['proton_energy']+ch['beta']['proton_energy']-ch['EC']['proton_energy'];el=ch['beta']['electron_energy']-ch['PC']['positron_energy']-ch['EC']['electron_energy'];dens=dict(EC=ec['proton_density']/(s.Ye*n)-1,PC=pc['neutron_density']/((1-s.Ye)*n)-1,beta=beta['parent_density']/((1-s.Ye)*n)-1)
 if any(not math.isfinite(v)or v<=0 for z in ch.values()for v in z.values())or max(abs(x)for x in dens.values())>L['gates']['source_density_relative']or max(abs((loss-Q)/Q),abs((en+ep+el+Q)/Q))>L['gates']['source_energy_relative']:raise ArithmeticError('Fresh physical source/density/energy gate failed')
 return dict(channels=ch,rates=rates,energy=energy,R=R,Q=Q,material_loss=loss,species_energy=dict(neutron=en,proton=ep,electron=el),density_residuals=dens,energy_residual=(en+ep+el+Q)/Q,direct_energy_residual=(Q-loss)/Q,profile=profile,controls=W.CONTROLS[profile])
def evaluate(t,x,Tguess):
 s,h=invert(t,float(x[0]),float(x[1]),Tguess);z=sources(s);n=s.nB;r=z['rates'];q=z['energy'];k=L['density']['k'];work=s.pressure*k/n;zeta=s.proton.chemical_potential+s.electron.chemical_potential-s.neutron.chemical_potential
 deriv=np.array([work-z['Q']/n,z['R']/n,q['EC']/n,(q['PC']+q['beta'])/n,r['EC']/n,(r['PC']+r['beta'])/n,work,(-z['Q']-zeta*z['R'])/(n*s.T),(r['beta']-r['EC'])/n,-r['PC']/n])
 Tdot=(((s.energy+s.pressure)/n-s.e_n)*(k*n)-z['Q']-s.e_Y*z['R']/n)/s.cv
 return deriv,dict(t=t,state=s.asdict(),inversion=h,source=z,derivative=dict(zip(NAMES,map(float,deriv))),Tdot=Tdot)
def run(steps):
 if type(steps)is not int or steps not in L['steps']:raise ValueError('Locked2/4/8 steps only')
 verify();beg=time.monotonic();dt=L['density']['duration']/steps;initial=M.state(**dict(T=L['initial']['T'],nB=L['initial']['nB'],Ye=L['initial']['Ye'],photons=True));x=np.zeros(len(NAMES));x[:2]=[initial.specific_energy,initial.Ye];guess=initial.T;records=[];endpoints=[]
 def save_partial():
  (P/f'partial_{steps}.json').write_text(json.dumps(dict(steps=steps,records=records,endpoints=endpoints,elapsed=time.monotonic()-beg),indent=2)+'\n')
 def bounded():
  if time.monotonic()-beg>L['budget_seconds_per_process']:raise TimeoutError('Preserved partial experiment budget stop')
 # Fresh forward anchor before interpreting any trajectory.
 d0,e0=evaluate(0.,x,guess);ref=json.loads((ROOT/'installed_free_matter_binding/weak_base.json').read_text());forward=[]
 for c in ['EC','PC','beta']:
  for f in ['number','neutrino_energy']:
   err=abs(e0['source']['channels'][c][f]/ref['source']['channels'][c][f]-1);forward.append(dict(name=c+f,error=err,passed=err<=L['gates']['initial_forward_relative']))
 if not all(v['passed']for v in forward):raise ArithmeticError('Initial absolute forward gate failed')
 endpoints.append(e0)
 try:
  for i in range(steps):
   bounded();t=i*dt;d1,a=evaluate(t,x,guess);bounded();xm=x+.5*dt*d1;dm,m=evaluate(t+.5*dt,xm,a['state']['T']);x+=dt*dm;bounded();dend,e=evaluate(t+dt,x,m['state']['T']);guess=e['state']['T'];records.append(dict(step=i,start=a,midpoint=m,final=e,conserved=dict(zip(NAMES,map(float,x)))));endpoints.append(e);save_partial()
 except Exception as err:
  save_partial();(P/f'failure_{steps}.json').write_text(json.dumps(dict(type=type(err).__name__,message=str(err),completed=len(records)),indent=2)+'\n');raise
 ef=M.state(guess,density(L['density']['duration']),float(x[1]),photons=True);escape=float(x[2]+x[3]);netlept=float(x[5]-x[4]);grosslept=float(x[5]+x[4]);sigma_delta=ef.entropy_per_baryon-initial.entropy_per_baryon
 emminus=ef.electron.particle_density/ef.nB-initial.electron.particle_density/initial.nB-x[8];emplus=ef.electron.antiparticle_density/ef.nB-initial.electron.antiparticle_density/initial.nB-x[9]
 times=np.array([e['t']for e in endpoints]);Qn=np.array([e['source']['Q']/e['state']['nB']for e in endpoints]);Rn=np.array([e['source']['R']/e['state']['nB']for e in endpoints]);Wn=np.array([e['state']['pressure']*L['density']['k']/e['state']['nB']for e in endpoints]);endpointQ=float(simpson(Qn,x=times));endpointR=float(simpson(Rn,x=times));endpointW=float(simpson(Wn,x=times));checks=[]
 def ck(name,error,gate):checks.append(dict(name=name,error=float(error),gate=gate,passed=bool(abs(error)<=gate)))
 g=L['gates'];ck('energy escape/work scaled',(ef.specific_energy-initial.specific_energy+escape-x[6])/max(escape,abs(x[6])),g['stage_energy_escape_or_work_scaled']);ck('electron Le gross scaled',(ef.Ye-initial.Ye-netlept)/grosslept,g['stage_lepton_gross_scaled']);ck('EM equal pairs',(emminus-emplus)/max(grosslept,abs(emminus),abs(emplus)),g['EMpair_gross_scaled']);ck('material entropy change',(sigma_delta-x[7])/abs(sigma_delta),g['material_entropy_change_scaled']);ck('fresh endpoint Simpson escape',(endpointQ-escape)/escape,g['independent_endpoint_quadrature_relative']);ck('fresh endpoint Simpson netlepton',(endpointR-(ef.Ye-initial.Ye))/grosslept,g['independent_endpoint_quadrature_relative']);ck('fresh endpoint Simpson pressurework',(endpointW-x[6])/abs(x[6]),g['independent_endpoint_quadrature_relative'])
 nonph0=(initial.energy-initial.photon_energy)/initial.nB;nonphf=(ef.energy-ef.photon_energy)/ef.nB;phdelta=ef.photon_energy/ef.nB-initial.photon_energy/initial.nB;ck('separate photon+nonphoton energy',(nonphf-nonph0+phdelta+escape-x[6])/escape,g['stage_energy_escape_or_work_scaled'])
 verify();out=dict(status='PASS'if all(c['passed']for c in checks)else'FAIL',steps=steps,initial=initial.asdict(),final=ef.asdict(),records=records,endpoints=endpoints,conserved=dict(zip(NAMES,map(float,x))),initial_forward=forward,checks=checks,passed=sum(c['passed']for c in checks)+sum(c['passed']for c in forward),failed=[c for c in checks if not c['passed']],independent_endpoints=dict(escape=endpointQ,net_lepton=endpointR,work=endpointW),EMpairs=dict(electron=float(emminus),positron=float(emplus)),material_entropy_delta=sigma_delta,photon_specific_energy_delta=phdelta,seconds=time.monotonic()-beg)
 (P/f'trajectory_{steps}.json').write_text(json.dumps(out,indent=2)+'\n');return out
if __name__=='__main__':
 z=run(int(sys.argv[1]));print(z['status'],z['passed'],z['failed'],'seconds',z['seconds'])
