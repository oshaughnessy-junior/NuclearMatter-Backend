"""Dedicated fresh-process entry; function modules only, never study drivers."""
from __future__ import annotations
import sys,math,importlib.util
from dataclasses import asdict
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import protocol as P

def main():
 pins=P.loads((P.HERE/'PINS.json').read_text());P.verify_pins(pins)
 raw=sys.stdin.read(200001)
 if len(raw)>200000:raise ValueError('Request too large')
 req=P.loads(raw);P.validate_request(req,pins)
 # Only in this disposable process may frozen source imports adjust sys.path.
 sys.path.insert(0,str(P.ROOT/'hot_beta_moments'));import hot_moments as H
 spec=importlib.util.spec_from_file_location('isolated_pc_invariant_function',P.ROOT/'review_pc_moments/reference.py');PC=importlib.util.module_from_spec(spec)
 P.verify_file(spec.origin,'review_pc_moments/reference.py',pins)
 sys.modules[spec.name]=PC;spec.loader.exec_module(PC)
 P.verify_file(PC.__file__,'review_pc_moments/reference.py',pins)
 # Audit every imported campaign source before any physical evaluation.
 for m in list(sys.modules.values()):
  f=getattr(m,'__file__',None)
  if f:
   p=Path(f).resolve()
   if p.is_relative_to(P.ROOT)and p.suffix=='.py'and str(p.relative_to(P.ROOT))not in pins['sources']:raise ValueError('Unpinned imported source '+str(p))
 s=req['state'];ec=req['controls']['eos']
 ns=H.K.eos.state_density(s['T'],s['nB']*(1-s['Ye']),H.K.MN,**ec)
 ps=H.K.eos.state_density(s['T'],s['nB']*s['Ye'],H.K.MP,**ec)
 es=H.K.eos.leptons.state_density(s['T'],s['nB']*s['Ye'],H.K.ME,rtol=ec['rtol'],tail=ec['tail'])
 bath=H.K.Bath(s['T'],ns.chemical_potential,ps.chemical_potential,es.chemical_potential)
 n,p,e=ns.asdict(),ps.asdict(),asdict(es)
 densities=dict(neutron=n['density'],proton=p['density'],electron=e['particle_density'],positron=e['antiparticle_density'])
 residual=max(abs(densities['neutron']/(s['nB']*(1-s['Ye']))-1),abs(densities['proton']/(s['nB']*s['Ye'])-1),abs((densities['electron']-densities['positron'])/(s['nB']*s['Ye'])-1))
 if residual>2e-9:raise ArithmeticError('Forward roots/neutrality gate failed')
 # Derivatives are accessed only after the unperturbed state passes.
 eos=dict(energy_density=sum(z['energy_density']for z in(n,p,e)),pressure=sum(z['pressure']for z in(n,p,e)),entropy_density=sum(z['entropy_density']for z in(n,p,e)),cv=n['cv_fixed_density']+p['cv_fixed_density']+e['cv_fixed_net_density'],densities=densities,total_mu=dict(neutron=bath.mu_n,proton=bath.mu_p,electron=bath.mu_e),masses=dict(neutron=H.K.MN,proton=H.K.MP,electron=H.K.ME),density_residual=residual)
 pc=req['controls']['pc'];bc=req['controls']['beta']
 a=PC.integral(vars(bath),order_n=pc[0],order_e=pc[1],order_angle=pc[2],tail=pc[3],budget=90.)
 b=H.integrate(bath,rest_order=bc[0],parent_order=bc[1],orient_order=bc[2],phi_order=bc[3],scenarios=[dict(name='empty',mode='empty')],budget_seconds=90.)
 pcR,pcQ=a['net'][0][:2];beta=b['scenarios']['empty']['net'];bR,bQ=beta['number'],beta['neutrino_energy'];R=pcR+bR;Q=pcQ+bQ
 residuals=dict(pc_density=a['neutron_density']/(s['nB']*(1-s['Ye']))-1,beta_density=b['parent_density']/(s['nB']*(1-s['Ye']))-1,energy=(a['net'][0][5]+beta['material_energy_loss']-Q)/Q)
 if max(abs(v)for v in residuals.values())>3e-8:raise ArithmeticError('Integrated forward source gate failed')
 pos=lambda v:P.represented(math.log(v))
 source=dict(channels=dict(positron_capture=dict(number=pos(pcR),energy=pos(pcQ)),neutron_beta=dict(number=pos(bR),energy=pos(bQ))),total=dict(number=pos(R),energy=pos(Q)),inverse=P.represented(None,zero_reason='empty_incident_radiation'),signed=dict(neutron=-R,proton=R,electron_net=R,antineutrino_e=R,material_energy=-Q,radiation_energy=Q,material_Le=R,radiation_Le=-R),residuals=residuals)
 P.verify_pins(pins)
 return dict(version=req['version'],status='ok',request=req,identity=P.digest(pins),runtime=P.runtime(),eos=eos,sources=source)
if __name__=='__main__':
 try:out=main();sys.stdout.write(P.dumps(out)+'\n')
 except Exception as ex:
  sys.stdout.write(P.dumps(dict(version='nmb-isolated-1',status='error',error_type=type(ex).__name__,message=str(ex)))+'\n');sys.exit(2)
