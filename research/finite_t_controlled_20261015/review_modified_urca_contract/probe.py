"""Read-only independent Fourier/Laplace and source/inventory MU contract probes."""
from __future__ import annotations
from pathlib import Path
import importlib.util,hashlib,json,math,time
from fractions import Fraction as F
from collections import Counter
import numpy as np
from scipy.integrate import quad
from scipy.special import expit,zeta
HERE=Path(__file__).resolve().parent;AUTHOR=HERE.parent/'modified_urca_contract'
checks=[];rows=[];start=time.monotonic()
def hashes():return {str(p.relative_to(AUTHOR)):hashlib.sha256(p.read_bytes()).hexdigest()for p in sorted(AUTHOR.rglob('*'))if p.is_file()and'__pycache__'not in str(p)}
before=hashes()
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
# ONLY the pure function module; no inventory or validation driver imports.
spec=importlib.util.spec_from_file_location('reviewed_mu_energy_pure',AUTHOR/'energy_reference.py');A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
# Exponential tilt: f(x)e^(x/2)/pi = 1/(2pi cosh(x/2)). Its
# characteristic function is sech(pi*k). FIVE independent convolutions
# follow by characteristic-function multiplication, not J2/J3/J5 polynomials.
def fourier5(y,L=10.):
 integral,err=quad(lambda k:math.cos(k*y)/math.cosh(math.pi*k)**5,0,L,epsabs=2e-14,epsrel=3e-13,limit=300)
 return math.exp(-y/2)*math.pi**4*integral
for y in [-21.3,-7.17,-.083,1.234,6.789,19.7]:
 v=fourier5(y);ref=A.J5(y)
 gate('new signed energy independent Fourier '+str(y),v/ref-1,2e-10)
 gate('Fourier cutoff8/10 '+str(y),fourier5(y,8.)/v-1,2e-10)
 rows.append(dict(y=y,fourier=v,author=ref))
for s in [.19,.413,.79]:
 val=quad(lambda y:math.exp(s*y)*A.J5(y),-250,250,epsabs=1e-8,epsrel=2e-12,points=[0],limit=300)[0]
 gate('independent bilateral Laplace '+str(s),val/(math.pi/math.sin(math.pi*s))**5-1,3e-12)
# ζ(2n)/pi^(2n) from the Bernoulli formula, exact rational arithmetic.
z4=F(1,90);z6=F(1,945);z8=F(1,9450)
coeff=(math.factorial(7)*F(127,128)*z8+10*math.factorial(5)*F(31,32)*z6+9*math.factorial(3)*F(7,8)*z4)/24
assert coeff==F(11513,120960)
ie=float(coeff)*math.pi**8
# Independently integrate the FOURIER phase kernel, never polynomial J5.
vals={}
for power,key in [(2,'number'),(3,'energy')]:
 # Finite upper y28 is controlled against y40 independently below.
 vals[key]=[]
 for tail in [28.,40.]:
  v=quad(lambda y:y**power*fourier5(y),0,tail,epsabs=1e-7,epsrel=2e-9,limit=150)[0];vals[key].append(v)
 exact=(float((1-2**-6)*math.factorial(6)*zeta(7)+10*math.pi**2*(1-2**-4)*math.factorial(4)*zeta(5)+9*math.pi**4*(1-2**-2)*math.factorial(2)*zeta(3))/24)if key=='number'else ie
 gate('Fourier integrated '+key,vals[key][-1]/exact-1,3e-8)
 # y28 intentionally coarse: record its error as diagnostic, not accept it.
 rows.append(dict(moment=key,tail28=vals[key][0],tail40=vals[key][1],exact=exact,tail28_relative_error=vals[key][0]/exact-1))
# Reconstruct inventory independently as stoichiometry, not author's Q dictionaries.
data=json.loads((AUTHOR/'inventory_results.json').read_text());charges={'n':(1,0,0,0),'p':(1,1,0,0),'em':(0,-1,1,0),'ep':(0,1,-1,0),'mum':(0,-1,0,1),'mup':(0,1,0,-1),'nue':(0,0,1,0),'anue':(0,0,-1,0),'numu':(0,0,0,1),'anumu':(0,0,0,-1)}
gate('32 unique directions',len({tuple((r[k]for k in ['flavor','spectator','family','direction']))for r in data['rows']})-32,0)
for r in data['rows']:
 label='/'.join(r[k]for k in['flavor','spectator','family','direction']);delta=Counter(r['final']);delta.subtract(r['initial'])
 for j,q in enumerate(['B','Q','Le','Lmu']):gate(label+' '+q,sum(v*charges[k][j]for k,v in delta.items()),0)
 fac=math.prod(math.factorial(v)for side in [r['initial'],r['final']]for v in Counter(side).values());gate(label+' independently counted identical pair',fac-2,0)
 inv=next(z for z in data['rows']if z['flavor']==r['flavor']and z['spectator']==r['spectator']and z['family']==r['family']and z['direction']!=r['direction']);gate(label+' exact reciprocal inventory',int(Counter(r['initial'])!=Counter(inv['final'])or Counter(r['final'])!=Counter(inv['initial'])),0)
# Five-Fermi polygon closure is independent of any EOS or matrix element.
def gap(p):return max(p)-(sum(p)-max(p))
for Y in [.009,.015,.016,.083,.12]:
 kn=(1-Y)**(1/3);kp=Y**(1/3);gl=gap([kn,kp,kp,kp,kp]);expected=Y>=1/65
 gate('electron-only proton MU polygon '+str(Y),int((gl<=0)!=expected),0)
 gate('polygon momentum unit conversion '+str(Y),gap([197.3269804*z for z in[kn,kp,kp,kp,kp]])/197.3269804-gl,3e-15)
# Spectral regulator normalization and W/2 convention, not a physical width model.
for W in [.031,2.7,19.]:
 M=637.;n=quad(lambda theta:1/math.pi,-math.pi/2,math.pi/2)[0];gate('signed-mass Lorentzian transform '+str(W),n-1,2e-14)
 for theta in [-1.13,.17,1.02]:
  m=M+W/2*math.tan(theta);R=(W/2)/(math.pi*((m-M)**2+(W/2)**2));jac=W/2/math.cos(theta)**2;gate('actual half-width Jacobian '+str((W,theta)),R*jac*math.pi-1,2e-11)
source=(AUTHOR/'source_snapshot/core_nu.py').read_bytes();blob=hashlib.sha1(b'blob '+str(len(source)).encode()+b'\0'+source).hexdigest();gate('pinned actual Git blob',int(blob!='520d3b458c5eddbed1e1e1a5c6270dcc12643eb5'),0)
text=source.decode();segment=text[text.index('def murca('):text.index('\ndef ',text.index('def murca(')+1)]
gate('source neutron electron+muon coefficient',int('(kfe / 1.68 + kfm / 1.68)'not in segment),0)
gate('source proton electron-only coefficient',int('mstp ** 3 * (kfe / 1.68)'not in segment),0)
gate('source coldT9 eighth power',int('t9 ** 8'not in segment),0)
gate('all frozen author bytes unchanged',int(hashes()!=before),0)
out=dict(checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],rows=rows,moments=vals,energy_coefficient=str(coeff),energy_integral=ie,mean_energy_over_T=vals['energy'][-1]/vals['number'][-1],source_blob=blob,author_hashes_before=before,author_hashes_after=hashes(),seconds=time.monotonic()-start,scope='contract, dimensionless five-degenerate-fermion phase only; no NN amplitude or absolute modified-Urca rate')
(HERE/'results.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:out[k]for k in['passed','failed','seconds','energy_coefficient','mean_energy_over_T']},indent=2))
