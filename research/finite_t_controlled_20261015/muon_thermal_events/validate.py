"""Bounded selected moving-muon event moment qualification and checkpoints."""
from pathlib import Path
import sys,json,time,math,hashlib
import numpy as np
from scipy.integrate import quad
from scipy.special import expit
sys.dont_write_bytecode=True
from muon_events import *
H=Path(__file__).parent;checks=[];rows=[];started=time.monotonic()
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and abs(error)<=tol)))
def save(phase):
 (H/'results.json').write_text(json.dumps(dict(passed=phase=='complete'and all(x['passed']for x in checks),phase=phase,checks=checks,rows=rows,runtime_seconds=time.monotonic()-started),indent=2)+'\n')
m=eos.matter_state(2.,1e-4,.29,.01);bath=Bath(2.,m.muon.chemical_potential,m.electron.chemical_potential);rad=Radiation(2.,15.,bath.mu_mu-bath.mu_e-15.-1.3)
for name,s,target in [('neutron',m.neutron,7e-5),('proton',m.proton,3e-5)]:gate(name+' EOS density',s.density/target-1,3e-10)
for name,s,target in [('electron',m.electron,2.9e-5),('muon',m.muon,1e-6)]:gate(name+' EOS net density',s.net_density/target-1,3e-10)
rows.append(dict(EOS=m.asdict(),radiation=asdict(rad)));save('EOS gated')
# Independent direct-momentum muon particle density and time-dilation moment.
end=math.sqrt((M+100*bath.T)**2-M*M)
f=lambda p:expit((bath.mu_mu-math.hypot(M,p))/bath.T)
npart=C*quad(lambda p:p*p*f(p),0,end,epsabs=1e-20,epsrel=3e-12)[0]
nproper=C*quad(lambda p:p*p*M/math.hypot(M,p)*f(p),0,end,epsabs=1e-20,epsrel=3e-12)[0]
gate('independent mu-minus momentum density',npart/m.muon.particle_density-1,3e-10)
rest=vacuum.pair_integral(40);rows.append(dict(anchors=dict(n_mu_minus=npart,n_mu_plus=m.muon.antiparticle_density,mean_m_over_E=nproper/npart,proper_vacuum=rest)))
settings=[dict(parent_order=12,pair_order=16,pair_angle=8,orientation_order=6,azimuth_order=8),dict(parent_order=20,pair_order=24,pair_angle=12,orientation_order=10,azimuth_order=16),dict(parent_order=28,pair_order=32,pair_angle=20,orientation_order=16,azimuth_order=24)]
for label,radiation,blocking in [('unblocked',None,False),('transparent',None,True),('FDpartners',rad,True)]:
 results=[]
 for controls in settings:
  out=integrate(bath,radiation,electron_blocking=blocking,**controls);results.append(out);rows.append(dict(label=label,result=out));save(label+' checkpoint'+str(len(results)))
 ref=results[-1]
 for i,coarse in enumerate(results[:-1]):
  for branch in ['forward','backward']:
   for key in ['number','electron_energy','antinue_energy','numu_energy','parent_energy']:
    a=coarse[branch][key]['value'];b=ref[branch][key]['value']
    gate(f'{label} refinement{i} {branch} {key}',a/b-1 if b else a,2e-4 if i==0 else 3e-6)
 for r in results:
  scale=r['forward']['number']['value']+r['backward']['number']['value'];norm=max(abs(r['net']['parent_energy']),r['forward']['parent_energy']['value']+r['backward']['parent_energy']['value'])
  gate(label+' per-event energy closure',r['max_event_energy_residual'],3e-13)
  gate(label+' independently accumulated total energy',(r['net']['parent_energy']-r['net']['electron_energy']-r['net']['antinue_energy']-r['net']['numu_energy'])/norm,3e-13)
  gate(label+' net vs separate gross',(r['net']['number']-(r['forward']['number']['value']-r['backward']['number']['value']))/scale,3e-13)
  ledger=r['ledgers'];gate(label+' total Le',(ledger['material_electron_number']-ledger['antinue_number'])/scale,1e-14);gate(label+' total Lmu',(ledger['material_muon_number']+ledger['numu_number'])/scale,1e-14);gate(label+' charge',(ledger['material_electron_number']+ledger['material_muon_number'])/scale,1e-14)
 if label=='unblocked':
  gate('unblocked moving time-dilated lifetime',ref['forward']['number']['value']/(nproper*rest['rate'])-1,3e-10)
  for key,restkey in [('electron_energy','energy_e'),('antinue_energy','energy_anti_e'),('numu_energy','energy_nu_mu')]:gate('unblocked Doppler '+key,ref['forward'][key]['value']/(npart*rest[restkey])-1,3e-10)
  gate('unblocked parent energy',ref['forward']['parent_energy']['value']/(npart*M*rest['rate'])-1,3e-10)
 if label=='FDpartners':
  affinity=(bath.mu_mu-bath.mu_e-rad.mu_anti_e-rad.mu_nu_mu)/bath.T
  for key in ['number','electron_energy','antinue_energy','numu_energy']:
   gate('independent F/B affinity '+key,ref['forward'][key]['log']-ref['backward'][key]['log']-affinity,5e-13)
  target=ref['net']['number']*affinity;gate('FD total entropy identity',ref['entropy']['total']/target-1,3e-13);gate('FD nonnegative total entropy',min(0.,ref['entropy']['total']),0.)
 # Separate rest-frame phase coordinate, not shared pair events.
 alt=integrate(bath,radiation,electron_blocking=blocking,phase='dalitz',parent_order=28,pair_order=32,pair_angle=20,orientation_order=16,azimuth_order=24)
 rows.append(dict(label=label+'_alternate_dalitz',result=alt));save(label+' alternate checkpoint')
 for branch in ['forward','backward']:
  for key in ['number','electron_energy','antinue_energy','numu_energy']:
   a=alt[branch][key]['value'];b=ref[branch][key]['value'];gate(label+' independent phase '+branch+' '+key,a/b-1 if b else a,3e-5)
 # Independently refine parent/phase/angle together, then tail separately.
 extended=integrate(bath,radiation,electron_blocking=blocking,parent_order=32,pair_order=40,pair_angle=24,orientation_order=20,azimuth_order=32,tail=70.)
 rows.append(dict(label=label+'_extended',result=extended));save(label+' extended checkpoint')
 for branch in ['forward','backward']:
  for key in ['number','electron_energy','antinue_energy','numu_energy']:
   a=extended[branch][key]['value'];b=ref[branch][key]['value'];gate(label+' extended '+branch+' '+key,a/b-1 if b else a,3e-6)
# FD common-T equilibrium, both separate positive events nonzero, stable local net.
LTE=Radiation(bath.T,15.,bath.mu_mu-bath.mu_e-15.)
q=integrate(bath,LTE,parent_order=20,pair_order=24,pair_angle=12,orientation_order=10,azimuth_order=16)
rows.append(dict(label='LTE',result=q));scale=q['forward']['number']['value']
gate('LTE positive events',float(scale<=0 or q['backward']['number']['value']<=0),0.);gate('LTE independently evaluated net',q['net']['number']/scale,3e-13);gate('LTE separate F/B',q['forward']['number']['value']/q['backward']['number']['value']-1,3e-13)
# Actual inverse sign: reverse common affinity, do not force event orientation.
reverse=Radiation(bath.T,15.,bath.mu_mu-bath.mu_e-15.+1.3);q=integrate(bath,reverse,parent_order=20,pair_order=24,pair_angle=12,orientation_order=10,azimuth_order=16)
rows.append(dict(label='inverse_dominant',result=q));gate('inverse dominated net sign',float(q['net']['number']>=0),0.);gate('inverse H theorem',min(0.,q['entropy']['total']),0.)
# Controls and unsupported mixed-temperature tail contract.
for name,fn in [('bad material T',lambda:Bath(-1,2,3)),('NaN material',lambda:Bath(2,float('nan'),3)),('bad radiation T',lambda:Radiation(0,1,2)),('negative tail',lambda:integrate(bath,tail=-1)),('boolean order',lambda:integrate(bath,parent_order=True)),('nonFD object',lambda:integrate(bath,{})),('bad phase',lambda:integrate(bath,phase='classical')),('hotter partner domain',lambda:integrate(bath,Radiation(3,1,2)))]:
 try:fn();rejected=False
 except ValueError:rejected=True
 gate(name,float(not rejected),0.)
save('complete');print(json.dumps(dict(passed=all(c['passed']for c in checks),checks=len(checks),failures=[c for c in checks if not c['passed']],runtime=time.monotonic()-started)))
