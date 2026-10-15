"""Read-only independent vacuum crossed-contract review; no author validators run."""
from pathlib import Path
import sys, json, hashlib, math, time
import numpy as np
from scipy.integrate import quad
from scipy.special import log_expit
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent; AUTHOR=ROOT/'relativistic_crossed_contract'
sys.path.insert(0,str(AUTHOR)); import beta as b
start=time.monotonic(); checks=[]; rows=[]
def hashes():
 paths=list(AUTHOR.glob('*'))+[ROOT/'relativistic_capture_oracle/events.py',ROOT/'weak_rates/static_capture.py']
 return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
initial=hashes()
def gate(name,error,tol):
 checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(np.isfinite(error) and abs(error)<=tol)))
def dot(x,y): return np.sum(x*y*np.array([1,-1,-1,-1]),axis=-1)
# Independent Weyl representation, not author's Dirac gamma or external spinors.
z=np.zeros((2,2),complex); one=np.eye(2); pauli=[np.array([[0,1],[1,0]]),np.array([[0,-1j],[1j,0]]),np.diag([1,-1])]
gamma=np.array([np.block([[z,one],[one,z]])]+[np.block([[z,s],[-s,z]]) for s in pauli]); g5=np.diag([-1,-1,1,1]); eye=np.eye(4); metric=np.array([1,-1,-1,-1]); ga=1.2723
GF=1.1663785e-11; VUD=.974; HBAR=6.582119569e-22
for mu in range(4):
 for nu in range(4):gate(f'Weyl Clifford {mu}{nu}',np.max(abs(gamma[mu]@gamma[nu]+gamma[nu]@gamma[mu]-2*(metric[mu] if mu==nu else 0)*eye)),0.)
def slash(p):return np.einsum('u,uij->ij',p*metric,gamma)
def barred(A):return gamma[0]@A.conj().T@gamma[0]
def trace_amp(pn,pp,l,k,mn,mp,ml):
 L=gamma@(eye-g5); H=gamma@(eye-ga*g5); total=0j
 for u in range(4):
  for v in range(4):
   lt=np.trace((slash(l)+ml*eye)@L[u]@slash(k)@barred(L[v]))
   ht=np.trace((slash(pp)+mp*eye)@H[u]@(slash(pn)+mn*eye)@barred(H[v]))
   total+=metric[u]*metric[v]*lt*ht
 return float(total.real*GF**2*VUD**2/4)
def scalar(pn,pp,l,k,mn,mp):
 # Trace-reduced crossing: no calibration to author's output.
 return 16*GF**2*VUD**2*((1+ga)**2*dot(pn,k)*dot(pp,l)+(1-ga)**2*dot(pn,l)*dot(pp,k)+(ga*ga-1)*mn*mp*dot(l,k))
def event(E,nu,mn,mp,ml):
 p=math.sqrt((E-ml)*(E+ml)); c=((mn-E)*nu-mn*((mn*mn-mp*mp+ml*ml)/(2*mn)-E))/(-p*nu)
 c=max(-1.,min(1.,c)); pn=np.array([mn,0.,0.,0.]); l=np.array([E,0.,0.,p]); k=np.array([nu,nu*math.sqrt((1-c)*(1+c)),0.,nu*c]);return pn,pn-l-k,l,k,c

def bounds(E,mn,mp,ml):
 p=math.sqrt(max(0.,(E-ml)*(E+ml))); Emax=((mn-mp)*(mn+mp)+ml*ml)/(2*mn); A=mn*(Emax-E); B=mn-E
 return A/(B+p), A/(B-p)

def dalitz(mn,mp,ml,which=0):
 Emax=((mn-mp)*(mn+mp)+ml*ml)/(2*mn)
 # E = ml + (Emax-ml)sin²x regularizes both endpoints. Independently
 # integrate neutrino energy rather than author electron-angle or pair-angle.
 def integrand(x):
  E=ml+(Emax-ml)*math.sin(x)**2; lo,hi=bounds(E,mn,mp,ml)
  def inner(nu):
   pn,pp,l,k,c=event(E,nu,mn,mp,ml); a=scalar(pn,pp,l,k,mn,mp)
   return a*([1.,nu,E][which])
  return quad(inner,lo,hi,epsabs=1e-35,epsrel=2e-11)[0]*2*(Emax-ml)*math.sin(x)*math.cos(x)
 return quad(integrand,0,math.pi/2,epsabs=1e-34,epsrel=2e-10)[0]/(64*math.pi**3*mn*HBAR)

for E,f in [(.543,.17),(.702,.82),(.973,.39),(1.271,.64)]:
 lo,hi=bounds(E,b.MN,b.MP,b.ME); nu=lo+f*(hi-lo); pn,pp,l,k,c=event(E,nu,b.MN,b.MP,b.ME)
 gate('independent event proton shell '+str(E),(dot(pp,pp)-b.MP*b.MP)/(b.MP*b.MP),6e-15)
 tr=trace_amp(pn,pp,l,k,b.MN,b.MP,b.ME); inv=scalar(pn,pp,l,k,b.MN,b.MP)
 args=[p[None,:] for p in [pn,pp,l,k]]; va=b.amplitude(*args,model='point_VA')[0]
 gate('independent Weyl vs invariant '+str(E),tr/inv-1,5e-13);gate('independent Weyl vs external crossed spinors '+str(E),va/tr-1,5e-13)
 full=b.amplitude(*args)[0]; reverse=b.amplitude(*args,inverse=True)[0]
 gate('full reciprocal amplitude '+str(E),reverse/full-1,5e-13)
 # Boost all external legs with independent Lorentz transform.
 vel=np.array([.23,-.41,.17]); v2=vel@vel; gm=1/math.sqrt(1-v2)
 def boost(p):
  vp=vel@p[1:];return np.r_[gm*(p[0]+vp),p[1:]+((gm-1)*vp/v2+gm*p[0])*vel]
 boosted=[boost(p)[None,:] for p in [pn,pp,l,k]]
 gate('full boost covariance '+str(E),b.amplitude(*boosted)[0]/full-1,2e-10)
 # Actual independent FD log-products, including a strongly degenerate state.
 for T,mup,mun,mue in [(1.13,937.,940.,.71),(.017,938.8,941.6,17.)]:
  lj=log_expit((mun-pn[0])/T)+log_expit((pp[0]-mup)/T)+log_expit((l[0]-mue)/T)
  lc=log_expit((mup-pp[0])/T)+log_expit((mue-l[0])/T)+log_expit((pn[0]-mun)/T)
  gate(f'anti KMS fresh E{E} T{T}',lj-lc+(k[0]-(mun-mup-mue))/T,3e-11)
 rows.append(dict(E=E,Enu=nu,c=c,trace=tr,scalar=inv,author_pointVA=va,author_full=full))

# Absolute Dalitz integration of invariant amplitude; no numerical spinors,
# author phase coordinates or author integration helpers on reference side.
reference=[dalitz(b.MN,b.MP,b.ME,i) for i in range(3)]
author=b.pair_integral(36,model='point_VA')
for key,val in zip(['rate','energy_neutrino','energy_electron'],reference):gate('independent Dalitz '+key,author[key]/val-1,3e-9)
rows.append(dict(independent_Dalitz=dict(zip(['rate','energy_neutrino','energy_electron'],reference)),author_pair36=author))
# Fresh differential electron spectra with direct antineutrino-energy integration.
for E in [.589,.863,1.157]:
 lo,hi=bounds(E,b.MN,b.MP,b.ME)
 def diff(nu):
  pn,pp,l,k,_=event(E,nu,b.MN,b.MP,b.ME)
  return scalar(pn,pp,l,k,b.MN,b.MP)/(64*math.pi**3*b.MN*HBAR)
 ref=quad(diff,lo,hi,epsabs=1e-16,epsrel=2e-11)[0]
 got=b.spectrum(E,28,model='point_VA')
 gate('direct Dalitz differential spectrum '+str(E),got/ref-1,3e-9)
 rows.append(dict(spectrum_E=E,Dalitz_spectrum=ref,author_pointVA_spectrum=got))
# Constant-amplitude phase measure: independent Dalitz area vs recursive
# two-body factorization, and analytic all-massless Gamma=M/(512*pi³).
for M,m,ml in [(b.MN,b.MP,b.ME),(5.,2.,.7),(3.,1.,0.),(4.,0.,0.)]:
 Emax=((M-m)*(M+m)+ml*ml)/(2*M)
 area=quad(lambda E:bounds(E,M,m,ml)[1]-bounds(E,M,m,ml)[0],ml,Emax,epsabs=1e-14,epsrel=2e-11)[0]
 gamma_d=area/(64*math.pi**3*M)
 def factor(s):
  if s==0:return 0.
  p=math.sqrt(max(0.,((M-m)**2-s)*((M+m)**2-s)))/(2*M)
  return p*(s-ml*ml)/(256*math.pi**3*M*M*s)
 gamma_s=quad(factor,ml*ml,(M-m)**2,epsabs=1e-17,epsrel=2e-10)[0]*2
 gate(f'absolute constant-amplitude phase M{M}',gamma_d/gamma_s-1,2e-9)
 if m==0 and ml==0:gate('analytic all-massless 3body phase',gamma_d/(M/(512*math.pi**3))-1,2e-12)
 rows.append(dict(constant_amplitude=dict(M=M,m=m,ml=ml,Dalitz=gamma_d,recursive=gamma_s)))
# Static normalization check by dimensionless analytic Fermi integral.
Q=b.MN-b.MP; W=Q/b.ME; r=math.sqrt(W*W-1)
f=(r*(2*W**4-9*W*W-8)+15*W*math.log(W+r))/60
static=GF**2*VUD**2*(1+3*ga*ga)*b.ME**5*f/(2*math.pi**3*HBAR)
old=json.loads((AUTHOR/'results.json').read_text()); static_author=next(x['static_rate'] for x in old['rows'] if 'static_rate' in x)
gate('analytic dimensionless static Fermi integral',static_author/static-1,3e-13)
rows.append(dict(static_rate_analytic=static,static_lifetime=1/static,exact_point_lifetime=1/reference[0]))
# Support follows invariant future-directed sum, not a rest-energy cut.
for mn,mp,ml,allowed in [(b.MN,b.MP,b.ME,True),(b.MN,b.MP,b.MMU,False),(b.MP,b.MN,b.ME,False)]:
 gate('invariant future sum support '+str((mn,mp,ml)),float((mn>=mp+ml)!=allowed),0.)
 if not allowed:
  try:b.pair_integral(8,mn,mp,ml); rejected=False
  except ValueError:rejected=True
  gate('forbidden vacuum branch rejected '+str(ml),float(not rejected),0.)
# No source/results mutations.
final=hashes();gate('all frozen author bytes unchanged',float(initial!=final),0.)
result=dict(passed=all(c['passed'] for c in checks),checks=checks,rows=rows,initial_hashes=initial,final_hashes=final,runtime_seconds=time.monotonic()-start)
(HERE/'results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(passed=result['passed'],checks=len(checks),failures=[c for c in checks if not c['passed']],runtime=result['runtime_seconds'])))
