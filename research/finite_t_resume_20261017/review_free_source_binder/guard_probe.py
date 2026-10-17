from pathlib import Path
from dataclasses import replace,asdict
from unittest.mock import patch
import sys,json,math,time
P=Path(__file__).resolve().parent;A=P.parent/'free_source_binder';sys.dont_write_bytecode=True;sys.path.insert(0,str(A/'staged/src'))
from nmbackend.reference import free_matter as F,weak_reference as W,_free_matter_sources as S
checks=[];start=time.monotonic()
def reject(name,fun,types=(ValueError,ArithmeticError,RuntimeError)):
 try:fun();checks.append(dict(name=name,passed=False,error='wrong acceptance'))
 except types as e:checks.append(dict(name=name,passed=True,error=type(e).__name__,message=str(e)))
s=F.state_density(1.7,8e-8,.27,photons='absent');calls=[]
def forbidden(*args,**kwargs):calls.append(args);raise AssertionError('Physical source must not run')
with patch.object(W,'channel_moments',side_effect=forbidden):
 for name,wrong in [('anti bound species',replace(s,anti_bounds=(replace(s.anti_bounds[0],species='positron'),s.anti_bounds[1]))),('muons retained',replace(s,muons_absent=False)),('wrong net density',replace(s,electron=replace(s.electron,net_density=s.electron.particle_density))),('wrong rest mass',replace(s,proton=replace(s.proton,mass=939.))),('photon identity',replace(s,photons='trapped_LTE')),('response forged',replace(s,denergy_dnB_fixed_T_Ye=s.denergy_dnB_fixed_T_Ye+1e-4))]:
  reject('canonical rejection before any rate: '+name,lambda wrong=wrong:F.source_moments(wrong,radiation=W.Radiation.empty(),controls=F.SourceControls.fine()))
 reject('new state wrong radiation T before rates',lambda:F.source_moments(s,radiation=W.Radiation.fd(1.700000001,0,0),controls=F.SourceControls.fine()))
 for val in [True,float('nan'),float('inf'),1e300]:
  reject('strictEOS T '+repr(val),lambda val=val:F.state_density(val,8e-8,.27,photons='absent'))
 # A prior same-process token never bypasses actual fresh canonical number roots.
 with patch.object(F.B,'state_density',side_effect=ArithmeticError('independent canonical root sentinel')):
  reject('canonical root failure precedes physical call',lambda:F.source_moments(s,radiation=W.Radiation.empty(),controls=F.SourceControls.fine()))
 for module in ['_weak_current.py','finite_t_leptons.py','constants.py']:
  orig=Path.read_bytes
  def changed(path):return orig(path)+(b'\n# independent synthetic stale bytes'if path.name==module else b'')
  with patch.object(Path,'read_bytes',changed):reject('stale '+module+' rejected before rates',lambda:F.source_moments(s,radiation=W.Radiation.empty(),controls=F.SourceControls.fine()))
checks.append(dict(name='all invalid source probes avoided physics',passed=not calls))
# Reconstruct actual saved raw channel records only for guard assembly; no physics qualification.
d=json.loads((A/'staged/tests/data/free_matter_reference_internal.json').read_text());rs=[]
for r in d['initial_channel_records']:
 rs.append(W.MomentResult(r['channel'],W.FreeElectronBath(**r['bath']),W.Radiation(**r['radiation']),W.Controls(**r['controls']),W.Current(),tuple(W.PositiveQuantity(**q)for q in r['forward']),tuple(W.PositiveQuantity(**q)for q in r['backward']),tuple(W.SignedQuantity(**q)for q in r['net']),None,r['max_event_kms_error'],tuple(tuple(x)for x in r['diagnostics']),tuple(tuple(x)for x in r['particle_stoichiometry']),r['energy_ledger_scaled'],r['direct_energy_ledger_scaled']))
s0=F.state_density(2,1e-7,.3,photons='absent')
with patch.object(W,'channel_moments',side_effect=rs):src=F.source_moments(s0,radiation=W.Radiation.empty(),controls=F.SourceControls.fine())
for name,wrong in [('receipt reconstructed',replace(src)),('gross source changed',replace(src,neutrino_energy_escape=src.material_energy_source)),('different flavor closure',replace(src,radiation=W.Radiation.fd(2,0,0))),('wrong complete inventory',replace(src,complete_channels=('EC','PC','PC')))]:
 reject('RHS rejects '+name,lambda wrong=wrong:F.instantaneous_rhs(s0,wrong,baryon_density_dot_fm3_s=0))
for val in [True,float('nan'),float('inf'),10**1000]:reject('strict nBdot '+str(val)[:20],lambda val=val:F.instantaneous_rhs(s0,src,baryon_density_dot_fm3_s=val))
# Underflow all6moments coherent: valid finite logs preserved; no reinterpretation as zero.
from nmbackend.reference._weak_quadrature import positive,signed
low=[replace(r,forward=tuple(positive(q.log-1500)for q in r.forward),net=tuple(signed(q.log_abs-1500,q.sign)for q in r.net))for r in rs]
with patch.object(W,'channel_moments',side_effect=low):under=F.source_moments(s0,radiation=W.Radiation.empty(),controls=F.SourceControls.fine())
checks.append(dict(name='finite-log aggregate underflow retained',passed=under.neutrino_energy_escape.value is None and math.isfinite(under.neutrino_energy_escape.log_abs)))
try:F.instantaneous_rhs(s0,under,baryon_density_dot_fm3_s=0);checks.append(dict(name='underflow RHS rejection',passed=False))
except S.UnrepresentedSourceError as e:checks.append(dict(name='underflow RHS rejection preserves exact source',passed=e.source is under))
(P/'GUARD_RESULTS.json').write_text(json.dumps(dict(checks=checks,passed=sum(x['passed']for x in checks),failed=[x for x in checks if not x['passed']],new_physical_rate_calls=0,scope='synthetic saved-record fault injection only',seconds=time.monotonic()-start),indent=2)+'\n')
print('GUARDS',sum(x['passed']for x in checks),'PASS',sum(not x['passed']for x in checks),'FAIL')
