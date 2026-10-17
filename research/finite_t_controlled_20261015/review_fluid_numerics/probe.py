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
 mu=m.electron.chemical_potential;me=a.physics.static.ME if hasattr(a.physics,'static')else .51099895
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
# Independently choose log(1+E) coordinate + extend cutoff; no author source helper.
def alt_source(T,Y,order):
 m=a.physics.matter_state(T,nB*(1-Y),nB*Y);xs,ws=leggauss(order);out={}
 for flavor in ['nue','antinue']:
  total=np.zeros(2)
  for lo,hi in zip([*a.EDGES,140.][:-1],[*a.EDGES,140.][1:]):
   zlo,zhi=math.log1p(lo),math.log1p(hi)
   for x,w in zip(xs,ws):
    if time.monotonic()-start>88:raise TimeoutError('Review90s experiment bound')
    z=zlo+(x+1)*(zhi-zlo)/2;E=math.expm1(z);weight=float(w)*(zhi-zlo)/2*math.exp(z)
    c=a.physics.coefficients(m,flavor,E,order=48,tail=40.)
    total+=weight/(2*math.pi**2*hc**3)*E**2*c['j_s']*np.array([1.,E])
  out[flavor]=total.tolist()
 return out
raw['endpoint_log12_cutoff140']=alt_source(final['T'],final['Ye'],12)
for f,val in raw['endpoint_log12_cutoff140'].items():
 for k,field in enumerate(['number_fm3_s','energy_MeV_fm3_s']):gate(f'endpoint log12 cutoff140 vs archived10 {f} {field}',abs(val[k]/report['final_refined_source']['moments'][f][field]-1),3e-6)
newT,newY=2.003,.297
raw['heldout_state']=dict(T=newT,Ye=newY,nB=nB)
raw['heldout_log8']=alt_source(newT,newY,8);raw['heldout_log12']=alt_source(newT,newY,12)
for f,val in raw['heldout_log12'].items():
 for k in range(2):gate(f'heldout log8->12 {f} moment{k}',abs(raw['heldout_log8'][f][k]/val[k]-1),3e-6)
endhash={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths}
result=dict(passed=all(c['passed']for c in checks)and hashes==endhash,checks=checks,raw=raw,runtime_seconds=time.monotonic()-start,source_sha256=hashes,end_source_sha256=endhash,scope='Selected independent EOS momentum integrals, ledgers, endpoint/new-state log-energy quadrature. No temporal-order qualification.')
(HERE/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=result['passed'],checks=len(checks),runtime=result['runtime_seconds'],failed=[c for c in checks if not c['passed']]),indent=2))
