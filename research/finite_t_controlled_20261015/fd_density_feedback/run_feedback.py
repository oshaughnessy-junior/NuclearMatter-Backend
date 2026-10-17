"""Bounded standalone2/4-step actual-source density-feedback experiments."""
from engine import *
import atexit
steps=int(sys.argv[1]);assert steps in (2,4)
start=time.monotonic();density_eos.set_budget(110.);provenance=physical_provenance();checks=[];rows=[];rates=[];failure=None
T0=2.;n0=1e-4;Y0=.3;duration=2e-6;dt=duration/steps
T=T0;n=n0;Y=Y0;s0=density_eos.state(T,n,Y);u0=s0['u_MeV_per_baryon'];escapedE=escapedL=work=entropy_weak=0.;work_positive=work_negative=0.
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error) and error<tol)))
def snapshot(phase):
 data=dict(phase=phase,steps=steps,checks=checks,rows=rows,sources=rates,failure=failure,provenance=provenance,runtime_seconds=time.monotonic()-start)
 (P/f'partial_{steps}.json').write_text(json.dumps(data,indent=2)+'\n')
atexit.register(lambda:snapshot('process_exit'))
try:
 for k in range(steps):
  snapshot('before_source')
  if time.monotonic()-start>92:raise TimeoutError('Too little budget for another physical source step')
  before=density_eos.state(T,n,Y);source=source_record(T,n,Y,provenance,initial=True if k==0 else None);rates.append(source);a=source['result']
  q=sum(a['moments'][flavor]['energy_MeV_fm3_s'] for flavor in ('nue','antinue'))/n
  F=(a['moments']['antinue']['number_fm3_s']-a['moments']['nue']['number_fm3_s'])/n
  gate(f'step{k} independent source energy assembly',abs(q/a['energy_loss_MeV_per_baryon_s']-1),1e-14)
  gate(f'step{k} independent source lepton assembly',abs(F/a['dYe_dt']-1),1e-14)
  uKick=before['u_MeV_per_baryon']-q*dt;Ykick=Y+F*dt
  if not 0<Ykick<1:raise ArithmeticError('Composition outside domain')
  Tkick=element.recover_temperature(uKick,n,Ykick,T);kick=density_eos.state(Tkick,n,Ykick)
  ds=weak_entropy_integral(before,uKick,Ykick,order=4);actual_ds=kick['entropy_per_baryon']-before['entropy_per_baryon']
  gate(f'step{k} weak-kick entropy firstlaw integral',abs(ds-actual_ds),2e-9)
  nextn=density_eos.density((k+1)/steps)
  nextT=density_eos.temperature_at_entropy(kick['entropy_per_baryon'],nextn,Ykick,Tkick)
  after=density_eos.state(nextT,nextn,Ykick)
  W=pressure_work(Tkick,n,Ykick,nextn,order=4)
  Wenergy=after['u_MeV_per_baryon']-uKick
  gate(f'step{k} independent mechanical pressurework',abs(W-Wenergy)/max(abs(W),1e-300),2e-8)
  gate(f'step{k} adiabatic entropy',abs(after['entropy_per_baryon']-kick['entropy_per_baryon']),2e-9)
  escapedE+=q*dt;escapedL-=F*dt;work+=W;entropy_weak+=ds
  work_positive+=max(W,0);work_negative+=min(W,0)
  gate(f'step{k} EOSmechanicalradiationenergyledger',abs(after['u_MeV_per_baryon']+escapedE-work-u0),3e-9)
  gate(f'step{k} electronleptonledger',abs(Ykick+escapedL-Y0),3e-13)
  gate(f'step{k} materialentropyledger',abs(after['entropy_per_baryon']-s0['entropy_per_baryon']-entropy_weak),3e-9)
  gate(f'step{k} netelectron neutrality',abs(after['matter'].electron.net_density/(nextn*Ykick)-1),2e-10)
  rows.append(dict(before=dict(T=T,nB=n,Ye=Y,u=before['u_MeV_per_baryon'],entropy=before['entropy_per_baryon']),weak_kick=dict(T=Tkick,nB=n,Ye=Ykick,u=uKick,entropy=kick['entropy_per_baryon']),after=dict(T=nextT,nB=nextn,Ye=Ykick,u=after['u_MeV_per_baryon'],entropy=after['entropy_per_baryon']),q=q,F=F,dt=dt,work_pressure=W,work_EOS_difference=Wenergy,weak_entropy_integral=ds,actual_weak_entropy_change=actual_ds))
  T,n,Y=nextT,nextn,Ykick;snapshot('step_complete')
except (ArithmeticError,ValueError,RuntimeError,TimeoutError) as exc:
 failure=dict(type=type(exc).__name__,message=str(exc));snapshot('failed_or_budget_stop')
endprovenance=physical_provenance();gate('exact physical source/hash/env provenance stable',float(provenance!=endprovenance),.1)
report=dict(passed=failure is None and len(rows)==steps and all(c['passed'] for c in checks),checks=checks,failure=failure,steps=steps,duration_s=duration,photons=False,method='First-order weak Euler kick then exactFDentropy density move; fresh source at every distinct state',initial=dict(T=T0,nB=n0,Ye=Y0,u=u0,entropy=s0['entropy_per_baryon']),final=dict(T=T,nB=n,Ye=Y,u=density_eos.state(T,n,Y)['u_MeV_per_baryon']),escaped_energy_per_baryon=escapedE,escaped_electron_lepton_per_baryon=escapedL,mechanical_work=work,compression_work=work_positive,expansion_work=work_negative,entropy_weak_integral=entropy_weak,rows=rows,sources=rates,provenance=provenance,end_provenance=endprovenance,runtime_seconds=time.monotonic()-start)
(P/f'results_{steps}.json').write_text(json.dumps(report,indent=2)+'\n');print(steps,sum(c['passed'] for c in checks),'/',len(checks),'passed',report['passed'],'runtime',report['runtime_seconds']);print('final',report['final'],'failure',failure)
raise SystemExit(0 if report['passed'] else 1)
