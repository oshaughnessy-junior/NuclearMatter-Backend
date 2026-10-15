"""Two/four isochoric Euler energy/Ye steps with new physical rates each state."""
from element import *
import json
P=Path(__file__).resolve().parent;qualified=json.loads((P/'source_qualification.json').read_text())
if not qualified['passed']:raise ValueError('Initial source qualification failed')
initial=qualified['coarse_source']
assert initial['T']==2. and initial['nB']==1e-4 and initial['Ye']==.3 and initial['outer_order']==6 and initial['internal_order']==48 and initial['tail']==40. and initial['edges']==list(EDGES)
start=time.monotonic();nB=1e-4;Y0=.3;T0=2.;duration=2e-6
memo={(T0,nB,Y0):initial};rate_evaluations=[];checks=[]
def check(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(error<tol)))
def rhs(T,Y):
 key=(T,nB,Y)
 if key not in memo:
  memo[key]=source(T,nB,Y,outer_order=6);rate_evaluations.append(memo[key])
 return memo[key]
def evolve(steps):
 T=T0;Y=Y0;u0=eos(T,nB,Y)['u_MeV_per_baryon'];u=u0;escapedE=0.;escapedL=0.;rows=[];dt=duration/steps
 for k in range(steps):
  rate=rhs(T,Y);old=dict(T=T,Ye=Y,u=u)
  # Evolve energy and conserved lepton charge; recover T from the full FD EOS.
  u-=dt*rate['energy_loss_MeV_per_baryon_s'];Y+=dt*rate['dYe_dt']
  escapedE+=dt*rate['energy_loss_MeV_per_baryon_s'];escapedL-=dt*rate['dYe_dt']
  if not 0<Y<1:raise ArithmeticError('Euler composition leaves domain')
  T=recover_temperature(u,nB,Y,T)
  actual=eos(T,nB,Y);energy_error=abs(actual['u_MeV_per_baryon']+escapedE-u0)/max(abs(u0),1.)
  lepton_error=abs(Y+escapedL-Y0)
  charge_error=abs(actual['matter'].electron.net_density/(nB*Y)-1)
  check(f'{steps}steps k{k} EOSradiationenergyledger',energy_error,3e-12)
  check(f'{steps}steps k{k} electronleptonledger',lepton_error,3e-13)
  check(f'{steps}steps k{k} forwardneutrality',charge_error,2e-10)
  rows.append(dict(before=old,after=dict(T=T,Ye=Y,u=u),dt=dt,rate_state=dict(T=rate['T'],Ye=rate['Ye']),q=rate['energy_loss_MeV_per_baryon_s'],dYe_dt=rate['dYe_dt'],escaped_energy_per_baryon=escapedE,escaped_electron_lepton_per_baryon=escapedL))
 return dict(steps=steps,rows=rows,final=dict(T=T,Ye=Y,u=u),escaped_energy_per_baryon=escapedE,escaped_electron_lepton_per_baryon=escapedL)
coarse=evolve(2);fine=evolve(4)
# Actual new-state quadrature controls at the final state, not a reuse of rates.
end6=rhs(fine['final']['T'],fine['final']['Ye']);end10=source(fine['final']['T'],nB,fine['final']['Ye'],outer_order=10)
for flavor in ('nue','antinue'):
 for key in ('number_fm3_s','energy_MeV_fm3_s'):
  check(f'new final state {flavor} outer6->10 {key}',abs(end6['moments'][flavor][key]/end10['moments'][flavor][key]-1),3e-6)
# Sample two weighted-relevant energies per flavor for internal48->72/tail50.
m=physics.matter_state(fine['final']['T'],nB*(1-fine['final']['Ye']),nB*fine['final']['Ye']);internal=[]
for flavor in ('nue','antinue'):
 for E in (4.,16.):
  a=physics.coefficients(m,flavor,E,order=48);b=physics.coefficients(m,flavor,E,order=72,tail=50.)
  for key in ('j_s','chi_s'):check(f'new final state internal {flavor} E{E} {key}',abs(a[key]/b[key]-1),3e-5)
  internal.append(dict(flavor=flavor,E=E,coarse=a,refined=b))
report=dict(passed=all(c['passed'] for c in checks),checks=checks,nB=nB,photons=False,duration_s=duration,method='Euler in u,Ye with genuine FD inversion; shared exact initial physical source, freshly recomputed at every distinct state',coarse=coarse,fine=fine,final_step_changes={k:fine['final'][k]-coarse['final'][k] for k in ('T','Ye','u')},fresh_physical_source_evaluations=rate_evaluations,final_refined_source=end10,final_internal_checks=internal,runtime_seconds=time.monotonic()-start,limits='Two/four short steps demonstrate local feedback and ledgers; no asymptotic trajectory certificate, interacting physics or hydro')
(P/'short_evolution_results.json').write_text(json.dumps(report,indent=2)+'\n');print(sum(c['passed'] for c in checks),'/',len(checks),report['runtime_seconds']);print('final',fine['final'],'stepdiff',report['final_step_changes'])
for c in checks:
 if not c['passed']:print('FAIL',c)
