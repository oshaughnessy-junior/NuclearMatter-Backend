from pathlib import Path
import sys,json,math,time,hashlib
import numpy as np
from scipy.integrate import quad
from scipy.special import expit
from numpy.polynomial.legendre import leggauss
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'fd_fluid_element'))
import element as a
start=time.monotonic();checks=[];raw={}
paths=[ROOT/'fd_fluid_element'/x for x in ['element.py','qualify_source.py','short_evolution.py','source_qualification.json','short_evolution_results.json']]
hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths}
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and error<tol)))
report=json.loads(paths[-1].read_text());nB=report['nB'];final=report['fine']['final'];hc=a.physics.HBARC
# Independent momentum-coordinate integrations, including charge, pairs and offsets.
def independent(T,Y):
 s=a.eos(T,nB,Y);m=s['matter'];values={};U=0.
 for name,sp in [('neutron',m.neutron),('proton',m.proton)]:
  end=math.sqrt(2*sp.mass*(max(sp.kinetic_mu,0)+100*T))
  def f(p):return expit((sp.kinetic_mu-p*p/(2*sp.mass))/T)
  density=quad(lambda p:p*p*f(p),0,end,epsabs=1e-12,epsrel=3e-12)[0]/(math.pi**2*hc**3)
  energy=quad(lambda p:p*p*(p*p/(2*sp.mass)+sp.offset)*f(p),0,end,epsabs=1e-10,epsrel=3e-12)[0]/(math.pi**2*hc**3)
  gate(f'FD momentum {T} {Y} {name} density',abs(density/sp.density-1),2e-10)
  U+=energy;values[name]=dict(density=density,energy=energy)
 mu=m.electron.chemical_potential;me=a.physics.static.ME if hasattr(a.physics,'static')else .51099895069
 end=abs(mu)+100*T+me
 em=lambda p:math.sqrt(p*p+me*me)
 charge=quad(lambda p:p*p*(expit((mu-em(p))/T)-expit((-mu-em(p))/T)),0,end,epsabs=1e-11,epsrel=3e-12)[0]/(math.pi**2*hc**3)
 ue=quad(lambda p:p*p*em(p)*(expit((mu-em(p))/T)+expit((-mu-em(p))/T)),0,end,epsabs=1e-10,epsrel=3e-12)[0]/(math.pi**2*hc**3)
 U+=ue
 gate(f'FD momentum {T} {Y} electron charge',abs(charge/(nB*Y)-1),2e-10)
 gate(f'FD momentum {T} {Y} total u',abs(U/nB/s['u_MeV_per_baryon']-1),2e-10)
 return U/nB,dict(species=values,electron_charge=charge,electron_energy=ue)
u0,_=independent(2.,.3);uf,detail=independent(final['T'],final['Ye']);raw['independent_endpoint']=detail
fine=report['fine'];gate('independent endpoint energy+escaped',abs(uf+fine['escaped_energy_per_baryon']-u0)/u0,2e-10)
gate('independent endpoint charge+escaped',abs(final['Ye']+fine['escaped_electron_lepton_per_baryon']-.3),2e-13)
# Audit each recorded source state/order and independently rebuild moments from panels.
for evolution in [report['coarse'],fine]:
 escapedE=escapedL=0.
 for k,row in enumerate(evolution['rows']):
  gate(f'{evolution["steps"]} row{k} source T state',abs(row['rate_state']['T']-row['before']['T']),1e-14)
  gate(f'{evolution["steps"]} row{k} source Y state',abs(row['rate_state']['Ye']-row['before']['Ye']),1e-14)
  escapedE+=row['dt']*row['q'];escapedL-=row['dt']*row['dYe_dt']
  gate(f'{evolution["steps"]} row{k} Euler energy ordering',abs(row['after']['u']-row['before']['u']+row['dt']*row['q']),2e-13)
  gate(f'{evolution["steps"]} row{k} Euler lepton ordering',abs(row['after']['Ye']-row['before']['Ye']-row['dt']*row['dYe_dt']),2e-13)
 gate(f'{evolution["steps"]} reconstructed energy counter',abs(escapedE-evolution['escaped_energy_per_baryon']),2e-13)
 gate(f'{evolution["steps"]} reconstructed lepton counter',abs(escapedL-evolution['escaped_electron_lepton_per_baryon']),2e-13)
for idx,src in enumerate(report['fresh_physical_source_evaluations']):
 rebuilt={f:np.sum([p['values']for p in src['panels']if p['flavor']==f],axis=0)for f in ['nue','antinue']}
 F=(rebuilt['antinue'][0]-rebuilt['nue'][0])/nB;q=(rebuilt['nue'][1]+rebuilt['antinue'][1])/nB
 gate(f'raw source{idx} lepton sign/DOS normalization',abs(F/src['dYe_dt']-1),1e-13)
 gate(f'raw source{idx} energy sign/normalization',abs(q/src['energy_loss_MeV_per_baryon_s']-1),1e-13)

endhash={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths}
result=dict(passed=all(c['passed']for c in checks)and hashes==endhash,checks=checks,raw=raw,runtime_seconds=time.monotonic()-start,source_sha256=hashes,end_source_sha256=endhash,source_quadrature_status='NOT_QUALIFIED: original88sprobe stopped during last heldout integral; unsaved integral arrays cannot serve as evidence',scope='Independent EOS momentum quadrature, charge and recorded raw-panel/source ordering/energy-lepton ledgers only; no new physical source quadrature claim or temporal-order certificate')
(HERE/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=result['passed'],checks=len(checks),runtime=result['runtime_seconds'],failed=[c for c in checks if not c['passed']]),indent=2))
