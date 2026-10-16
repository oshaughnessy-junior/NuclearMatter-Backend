"""Fresh -I -B physical worker for selected photon+npe EOS / three CC sources.

No driver imports, files written, rate cache or external thermostat.
"""
from __future__ import annotations
from pathlib import Path
import json,math,sys,hashlib,importlib.util,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;NB=1e-7
CONTROLS={
 'base':dict(EC=[20,32,8,50.],PC=[28,44,10,60.],beta=[16,20,8,12,'pair','laguerre',60.]),
 'fine':dict(EC=[28,48,12,60.],PC=[32,52,12,60.],beta=[24,28,10,16,'electron','direct_p',80.]),
 'tail':dict(EC=[28,48,12,80.],PC=[32,52,12,80.],beta=[24,28,10,16,'electron','direct_p',100.])}
def numeric(v):return type(v)in(int,float)and math.isfinite(v)
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def rejectconstant(x):raise ValueError('Nonfinite JSON constant')
def loads(s):return json.loads(s,parse_constant=rejectconstant)
REQUIRED_FILES=['crossed_thermal_moments/moments.py', 'crossed_thermal_moments/spectrum_anchor.py', 'hot_beta_moments/hot_moments.py', 'hot_electron_capture_moments/electron_capture.py', 'muon_full_current/current.py', 'muon_vacuum/vacuum_cc.py', 'positron_capture_moments/moments.py', 'relativistic_capture_oracle/events.py', 'relativistic_crossed_contract/beta.py', 'relativistic_crossed_thermal/kernel.py', 'relativistic_free_eos/free_baryons.py', 'relativistic_tensor/tensor.py', 'review_hot_electron_capture/CHECKPOINT.json', 'review_hot_electron_capture/REVIEW.md', 'review_pc_moments/reference.py', 'review_transparent_capture_balance/CHECKPOINT.json', 'review_transparent_capture_balance/REVIEW.md', 'thermodynamics/lepton_reference.py', 'three_source_cooling_element/client.py', 'three_source_cooling_element/worker.py', 'transparent_capture_balance/EC_REVIEW_ACCEPTANCE.json', 'transparent_capture_balance/UPSTREAM.json', 'transparent_capture_balance/balance.py']
EXPECTED_LOADED=['hot_beta_moments/hot_moments.py', 'hot_electron_capture_moments/electron_capture.py', 'muon_full_current/current.py', 'muon_vacuum/vacuum_cc.py', 'relativistic_capture_oracle/events.py', 'relativistic_crossed_contract/beta.py', 'relativistic_crossed_thermal/kernel.py', 'relativistic_free_eos/free_baryons.py', 'relativistic_tensor/tensor.py', 'review_pc_moments/reference.py', 'thermodynamics/lepton_reference.py', 'three_source_cooling_element/worker.py', 'transparent_capture_balance/balance.py']
def validate_pin_record(p):
 if p.get('accepted_balance')is not True or set(p.get('files',{}))!=set(REQUIRED_FILES)or sorted(p.get('expected_loaded_python',[]))!=EXPECTED_LOADED:raise ValueError('Exact mandatory source/review set and literal acceptance required')
 return p

def pins():
 p=validate_pin_record(loads((HERE/'PINS.json').read_text()))
 review=loads((ROOT/'review_transparent_capture_balance/CHECKPOINT.json').read_text())
 if not isinstance(review.get('status'),str)or not review['status'].startswith('ACCEPTED')or review.get('frozen_dependencies',{}).get('transparent_capture_balance/balance.py')!=p['files']['transparent_capture_balance/balance.py']:raise RuntimeError('Source-matched accepted balance review required')
 for rel,h in p['files'].items():
  q=ROOT/rel
  if not q.is_file()or hashlib.sha256(q.read_bytes()).hexdigest()!=h:raise RuntimeError('Frozen source/acceptance changed: '+rel)
 return p

def request(q,p):
 if set(q)!=set(['action','T','Ye','nB','photons','profile','identity']):raise ValueError('Exact request fields required')
 if q['action']not in['state','source']or type(q['photons'])is not bool or not all(numeric(q[k])for k in['T','Ye','nB'])or not 1.8<=q['T']<=2.1 or not .25<=q['Ye']<=.4 or q['nB']!=NB:raise ValueError('Selected fixed-nB warm material admission only')
 if q['profile']not in CONTROLS or q['identity']!=digest(p):raise ValueError('Exact source/control identity required')
 return q

def physical():
 spec=importlib.util.spec_from_file_location('three_source_frozen_balance_functions',ROOT/'transparent_capture_balance/balance.py');B=importlib.util.module_from_spec(spec);sys.modules[spec.name]=B;spec.loader.exec_module(B);B.require_acceptance();return B

def loaded_sources():
 paths={}
 for mod in list(sys.modules.values()):
  f=getattr(mod,'__file__',None)
  if f:
   p=Path(f).resolve()
   if p.suffix=='.py'and p.is_relative_to(ROOT):paths[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 return paths

def state(B,T,Ye,ph):
 bath,raw=B.EC.matched(T,NB,Ye);n,p,e=[raw[k]for k in['neutron','proton','electron']];HC=B.EC.HC
 energy=sum(z['energy_density']for z in[n,p,e]);cv=n['cv_fixed_density']+p['cv_fixed_density']+e['cv_fixed_net_density'];pressure=sum(z['pressure']for z in[n,p,e]);entropy=sum(z['entropy_density']for z in[n,p,e])
 ae=lambda z,k:z['chemical_potential']-T*z[k]
 uy=ae(p,'dmu_dtemperature_fixed_density')-ae(n,'dmu_dtemperature_fixed_density')+ae(e,'dmu_dtemperature_fixed_net_density')
 eg=math.pi**2*T**4/(15*HC**3)if ph else 0.;pg=eg/3;sg=4*eg/(3*T);cg=4*eg/T;charge=e['particle_density']-e['antiparticle_density'];res=max(abs(n['density']/(NB*(1-Ye))-1),abs(p['density']/(NB*Ye)-1),abs(charge/(NB*Ye)-1))
 if res>2e-9 or cv<=0:raise ArithmeticError('Fresh FD forward roots/Cv failed')
 return dict(T=T,Ye=Ye,nB=NB,photons=ph,u=(energy+eg)/NB,energy_density=energy+eg,cv=cv+cg,u_T=(cv+cg)/NB,u_Y=uy,pressure=pressure+pg,entropy_density=entropy+sg,photon=dict(energy_density=eg,pressure=pg,entropy_density=sg,cv=cg,chemical_potential=0.,polarizations=2,transport='trapped LTE; no photon escape'),density_residual=res,bath=vars(bath),material=raw,electron_particle_density=e['particle_density'],positron_density=e['antiparticle_density'])

def source(B,s,profile):
 start=time.monotonic();b=s['bath'];bath=B.EC.Bath(**b);ecp=CONTROLS[profile]['EC'];pcp=CONTROLS[profile]['PC'];bp=CONTROLS[profile]['beta']
 ec=B.EC.integrate(bath,proton_order=ecp[0],electron_order=ecp[1],initial_order=ecp[2],outgoing_order=ecp[2],phi_order=ecp[2],tail=ecp[3],scenarios=[dict(name='empty',mode='empty')],budget_seconds=90.)
 pc=B.PC.integral(b,order_n=pcp[0],order_e=pcp[1],order_angle=pcp[2],tail=pcp[3],budget=90.)
 beta=B.BETA.integrate(B.BETA.K.Bath(**b),rest_order=bp[0],parent_order=bp[1],orient_order=bp[2],phi_order=bp[3],coordinate=bp[4],parent_coordinate=bp[5],tail=bp[6],scenarios=[dict(name='empty',mode='empty')],budget_seconds=90.)
 channels=dict(EC=ec['scenarios']['empty']['net'],PC=dict(zip(B.PC.NAMES,map(float,pc['sums'][0][0]))),beta=beta['scenarios']['empty']['net']);rates={k:v['number']for k,v in channels.items()};q={k:v['neutrino_energy']for k,v in channels.items()};R=rates['beta']+rates['PC']-rates['EC'];Q=sum(q.values());loss=sum(z['material_energy_loss']for z in channels.values());mEn=channels['EC']['neutron_energy']-channels['beta']['neutron_energy']-channels['PC']['neutron_energy'];mEp=channels['beta']['proton_energy']+channels['PC']['proton_energy']-channels['EC']['proton_energy'];mEl=channels['beta']['electron_energy']-channels['PC']['positron_energy']-channels['EC']['electron_energy'];dens=dict(EC=ec['proton_density']/(NB*s['Ye'])-1,PC=pc['neutron_density']/(NB*(1-s['Ye']))-1,beta=beta['parent_density']/(NB*(1-s['Ye']))-1)
 if any(not numeric(v)or v<=0 for z in channels.values()for v in z.values())or max(abs(v)for v in dens.values())>3e-8 or abs((loss-Q)/Q)>2e-10 or abs((mEn+mEp+mEl+Q)/Q)>2e-10:raise ArithmeticError('Fresh channel forward/density/energy gate failed')
 return dict(profile=profile,controls=CONTROLS[profile],channels=channels,R=R,Q=Q,rates=rates,energy=q,direct_loss=loss,material_energy_from_species=mEn+mEp+mEl,direct_energy_ledger=(Q-loss)/Q,species_energy_ledger=(mEn+mEp+mEl+Q)/Q,density_residuals=dens,number_sources=dict(n=-R,p=R,electron_net=R,electron_particle=rates['beta']-rates['EC'],positron=-rates['PC'],nue=rates['EC'],antinue=rates['beta']+rates['PC']),u_dot=-Q/NB,Ye_dot=R/NB,T_dot=(-Q/NB-s['u_Y']*R/NB)/s['u_T'],raw=dict(EC=ec,PC=pc,beta=beta),seconds=time.monotonic()-start)

def main():
 started=time.monotonic();p=pins();q=request(loads(sys.stdin.read()),p);B=physical();loaded=loaded_sources()
 if set(loaded)!=set(p['expected_loaded_python'])or any(rel not in p['files']or p['files'][rel]!=h for rel,h in loaded.items()):raise RuntimeError('Missing or unpinned loaded campaign dependency')
 s=state(B,q['T'],q['Ye'],q['photons']);out=dict(request=q,identity=digest(p),state=s,loaded_sources=loaded,sources=source(B,s,q['profile'])if q['action']=='source'else None,elapsed=0.)
 if loaded_sources()!=loaded:raise RuntimeError('Late campaign import changed dependency tree')
 pins();out['elapsed']=time.monotonic()-started;print(json.dumps(out,allow_nan=False,separators=(',',':')))
if __name__=='__main__':
 try:main()
 except Exception as e:print(json.dumps(dict(error=type(e).__name__,message=str(e))));raise SystemExit(1)
