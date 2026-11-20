"""Independent momentum-static moments and logarithmic-energy physical quadrature."""
from pathlib import Path
import sys,math,json,time,hashlib
import numpy as np
from scipy.special import expit,logsumexp
from scipy.integrate import quad
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'emission_only';OUT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True;sys.path.insert(0,str(A));import log_emission as r
pinned=hashlib.sha256((A/'log_emission.py').read_bytes()).hexdigest();assert pinned.startswith('f73fd350')
files=[A/'log_emission.py',A/'initial_import_failure.txt',A/'initial_representation_serialization_failure.txt',A/'initial_partial_before_flag_cast.json',ROOT/'quantum_energy_moments/static_inventory_failure.json']
hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
start=time.monotonic();checks=[]
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error) and error<tol)))
b=r.make_bath(.3,7e-5,3e-5,1e7,0.);T=b.T;me=r.static.ME;Q=r.static.Q
for name,sp in [('neutron',b.neutron),('proton',b.proton)]:
 val=quad(lambda x:math.sqrt(x)*expit(sp.kinetic_mu/T-x),0,100,epsabs=1e-17,epsrel=1e-12)[0]*(2*sp.mass*T)**1.5/(2*math.pi**2*r.static.HBARC**3)
 gate('forward independent dimensionless kinetic density '+name,abs(val/sp.density-1),2e-10)
gate('prescribed electron bath not neutralized',abs(b.mu_e),1e-15)
gate('physical Q offset retained',abs((b.neutron.offset-b.proton.offset)-Q),1e-15)
# Independent electron MOMENTUM integration. No target_moments, spectral,
# author endpoint map or energy quadrature used for these static references.
GF=1.1663785e-11;Vud=.974;gA=1.2723;hbar=6.582119569e-22
pref=GF**2*Vud**2*(1+3*gA*gA)/(2*math.pi**3*hbar)
static={}
for channel in r.CHANNELS:
 vals=[]
 for k in (0,1):
  def fn(p):
   Ee=math.hypot(p,me)
   if channel=='positron_capture':Enu=Ee+Q;block=expit((-Ee-b.mu_e)/T)
   else:Enu=Q-Ee;block=expit((Ee-b.mu_e)/T)
   return p*p*Enu**(2+k)*block
  upper=math.inf if channel=='positron_capture' else math.sqrt(Q*Q-me*me)
  val,err=quad(fn,0,upper,epsabs=1e-14,epsrel=2e-12,limit=200)
  vals.append(pref*b.neutron.density*val)
 static[channel]=vals
static['combined']=[sum(static[c][k] for c in r.CHANNELS) for k in (0,1)]
author=json.loads((A/'results.json').read_text());fine=author['runs'][-1]
for channel in (*r.CHANNELS,'combined'):
 for k,kind in enumerate(('number','energy')):
  gate(f'author heavyM versus independent p-static {channel} {kind}',abs(math.expm1(fine['log_moments'][channel][k]-math.log(static[channel][k]))),3e-5)
# New selected allowed points, both named channels always computed.
heldout=[]
for E,channel in [(.23,'neutron_decay'),(2.6,'positron_capture')]:
 logs=r.log_spectra(b,E,energy_order=32,angle_order=32,tail=60.)
 Ee=(Q-E) if channel=='neutron_decay' else (E-Q);pe=math.sqrt((Ee-me)*(Ee+me))
 block=expit((Ee-b.mu_e)/T) if channel=='neutron_decay' else expit((-Ee-b.mu_e)/T)
 expected=pref/r.DOS*b.neutron.density*Ee*pe*block
 gate(f'heldout heavyM spectral {channel} E{E}',abs(math.expm1(logs[channel]-math.log(expected))),3e-5)
 gate(f'heldout log inventory sum E{E}',abs(logs['combined']-float(logsumexp([logs[c] for c in r.CHANNELS]))),1e-13)
 heldout.append(dict(E=E,logs=logs,static_expected=expected))
gap=r.log_spectra(b,.8313293913178066);representations={c:r.represented(v) for c,v in gap.items()}
for c in (*r.CHANNELS,'combined'):
 gate('gap finite negative log '+c,float(not math.isfinite(gap[c]) or gap[c]>=0),.1)
 gate('gap None and explicit underflow '+c,float(not(representations[c]['value'] is None and representations[c]['reason']=='underflow' and not representations[c]['representable'])),.1)
gate('gap ND retains extreme finite tail',abs(gap['neutron_decay']-author['gap_witness']['neutron_decay']['log_value']),1e-8)
# New full physical energy integration: log(E) coordinate, geometric threshold
# panels and ordinary linear low-E panel. It neither omits the gap nor applies
# a static support cutoff. All accumulation remains logarithmic.
thresholds=[Q-me,Q+me];edges={0.,1e-5,.1,.4,.65,1.1,1.5,2.,3.,5.,8.,12.,20.,32.}
for e in thresholds:
 width=e*math.sqrt(T/b.neutron.mass)
 edges.add(e)
 for n in [1,4,16,64,256]:edges.update([e-n*width,e+n*width])
edges=sorted(x for x in edges if 0<=x<=32)
def integrate(order):
 xs,ws=np.polynomial.legendre.leggauss(order);terms={c:[[],[]] for c in r.CHANNELS}
 for lo,hi in zip(edges[:-1],edges[1:]):
  if lo==0:
   energies=.5*(xs+1)*hi;weights=.5*hi*ws
   logmeasure=np.log(weights)
  else:
   a,z=math.log(lo),math.log(hi);y=a+.5*(xs+1)*(z-a)
   energies=np.exp(y);logmeasure=np.log(.5*(z-a)*ws)+y
  for E,lw in zip(energies,logmeasure):
   if time.monotonic()-start>80:raise TimeoutError('Independent numerical budget bound reached')
   logs=r.log_spectra(b,float(E),energy_order=24,angle_order=24,tail=40.)
   for c in r.CHANNELS:
    for k in (0,1):terms[c][k].append(math.log(r.DOS)+float(lw)+(2+k)*math.log(float(E))+logs[c])
 ans={c:[float(logsumexp(terms[c][k])) for k in (0,1)] for c in r.CHANNELS}
 ans['combined']=[float(logsumexp([ans[c][k] for c in r.CHANNELS])) for k in (0,1)]
 return ans
coarse=integrate(10);refined=integrate(16)
for c in (*r.CHANNELS,'combined'):
 for k,kind in enumerate(('number','energy')):
  gate(f'independent logE10->16 {c} {kind}',abs(math.expm1(coarse[c][k]-refined[c][k])),3e-6)
  gate(f'independent logE16 versus author mapped20 {c} {kind}',abs(math.expm1(refined[c][k]-fine['log_moments'][c][k])),3e-6)
  gate(f'independent logE16 versus staticp {c} {kind}',abs(math.expm1(refined[c][k]-math.log(static[c][k]))),3e-5)
final={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
gate('author source/failures unchanged',float(hashes!=final),.1)
result=dict(status='PASS' if all(c['passed'] for c in checks) else 'FAIL',checks=checks,pinned_log_emission_sha256=pinned,static_momentum_integrals=static,heldout=heldout,gap_logs=gap,gap_representations=representations,alternate_logE_edges=edges,alternate_coarse=coarse,alternate_refined=refined,source_hashes=hashes,end_hashes=final,runtime_seconds=time.monotonic()-start,limits='Single prescribed-electron heavyM normalization bath; shared finiteM response, no extreme-tail relative accuracy, ordinaryopacity/transfer or oldT2wrapper qualification')
(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(result['status'],sum(c['passed'] for c in checks),'/',len(checks),result['runtime_seconds'])
for c in checks:
 if not c['passed']:print('FAIL',c)
