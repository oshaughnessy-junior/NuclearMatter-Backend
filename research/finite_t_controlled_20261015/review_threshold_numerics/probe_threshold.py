"""Independent forward-state, held-out angular/energy, and cold-support review."""
from pathlib import Path
import sys,math,json,hashlib,time
import numpy as np
from scipy.integrate import quad
from scipy.special import expit,log_expit,logsumexp
ROOT=Path(__file__).resolve().parents[1];AUTHOR=ROOT/'urca_threshold';OUT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(AUTHOR));import threshold_study as author
sourced=AUTHOR/'threshold_study.py';sourcehash=hashlib.sha256(sourced.read_bytes()).hexdigest()
assert sourcehash=='d658eceaf83574971940540fdc7f413cec47df6ba7ce0f955d7c1f5a9b787d1a'
start=time.monotonic();checks=[]
def check(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error) and error<tol)))
D=json.loads((AUTHOR/'qualified_results.json').read_text());rows=D['rows'];fail=json.loads((AUTHOR/'initial_cold_extension_failure.json').read_text())
check('qualified inventory nine',abs(len(rows)-9),.1)
check('qualified points exclude T=.01',float(any(r['T']==.01 for r in rows)),.1)
check('excluded cold capture failures preserved',abs(len(fail['failures'])-2),.1)
check('qualified source contains same pinned driver',float(hashlib.sha256(sourced.read_bytes()).hexdigest()!=sourcehash),.1)
# Independent energy-coordinate FD moment quadrature (author forward/root uses
# other coordinates). Split at kinetic Fermi energy, not root integrator nodes.
def nr_density(sp):
 mu=sp.kinetic_mu;T=sp.T;upper=max(mu,0)+100*T
 edges=sorted({0.,upper}|{x for x in [mu-30*T,mu-6*T,mu,mu+6*T,mu+30*T] if 0<x<upper})
 val=sum(quad(lambda e:math.sqrt(e)*expit((mu-e)/T),lo,hi,epsabs=1e-13,epsrel=1e-12)[0] for lo,hi in zip(edges[:-1],edges[1:]))
 return (2*sp.mass)**1.5/(2*math.pi**2*author.eos.HBARC**3)*val

def electron_density(el):
 T=el.temperature;mu=el.chemical_potential;m=el.mass;upper=max(abs(mu),m)+100*T
 edges=sorted({m,upper}|{x for x in [mu-30*T,mu-6*T,mu,mu+6*T,mu+30*T] if m<x<upper})
 val=sum(quad(lambda E:E*math.sqrt((E-m)*(E+m))*(expit((mu-E)/T)-expit((-mu-E)/T)),lo,hi,epsabs=1e-12,epsrel=1e-12)[0] for lo,hi in zip(edges[:-1],edges[1:]))
 return val/(math.pi**2*author.eos.HBARC**3)
for row in rows:
 ref=row['reference'];Y=ref['Yp'];T=row['T'];m=author.make_state(ref,T)
 # Reconstruct the per-Y cold calibration directly from Fermi momenta.
 pn=author.eos.HBARC*(3*math.pi**2*.01*(1-Y))**(1/3);pp=author.eos.HBARC*(3*math.pi**2*.01*Y)**(1/3)
 delta=(pp*pp-pn*pn)/(2*author.eos.MASS)+math.hypot(pp,author.a.ME)
 check(f'fixed cold calibration Y{Y} T{T}',abs(ref['offset_neutron']-delta),2e-12)
 for name,value,target in [('nn',nr_density(m.neutron),.01*(1-Y)),('np',nr_density(m.proton),.01*Y),('ne_net',electron_density(m.electron),.01*Y)]:check(f'forward independent energy density {name} Y{Y} T{T}',abs(value/target-1),2e-9)
 for key,actual in [('mu_n',m.neutron.mu),('mu_p',m.proton.mu),('mu_e',m.electron.chemical_potential)]:check(f'archived forward {key} Y{Y} T{T}',abs(actual-row[key]),2e-12)
# New Y,T AND E/T; same offset calibration is fixed before changing T.
Y=.105;T=.047;Enu=2.7*T
pn=author.eos.HBARC*(3*math.pi**2*.01*(1-Y))**(1/3);pp=author.eos.HBARC*(3*math.pi**2*.01*Y)**(1/3)
M=author.eos.MASS;delta=(pp*pp-pn*pn)/(2*M)+math.hypot(pp,author.a.ME)
m=author.eos.matter_state(T,.01*(1-Y),.01*Y,offset_neutron=delta,offset_proton=0.)
for name,value,target in [('nn',nr_density(m.neutron),.01*(1-Y)),('np',nr_density(m.proton),.01*Y),('ne_net',electron_density(m.electron),.01*Y)]:check(f'heldout forward independent density {name}',abs(value/target-1),2e-9)
opt=m.optical('nue');fine=author.guarded_capture.log_coefficients(opt,Enu,energy_order=72,angle_order=72,tail=60.)
# Independent adaptive electron energy in y and direct cos(theta) Gauss rule.
# The author independent comparator uses q rather than cos(theta).
def adaptive_log_chi(order,tail):
 cs,cws=np.polynomial.legendre.leggauss(order);mu=opt.mu_lepton;low=(author.a.ME-mu)/T
 edges=sorted({low,tail}|{v for v in [-120.,-60.,-30.,-10.,-3.,0.,3.,10.,30.,60.,120.] if low<v<tail})
 def logfn(y):
  Ee=mu+T*y
  if Ee<=author.a.ME:return -math.inf
  pe=math.sqrt((Ee-author.a.ME)*(Ee+author.a.ME));q=np.sqrt((Enu-pe)**2+2*Enu*pe*(1-cs))
  vals=[math.log(float(w)*Ee*pe*author.a.angular_factor(Ee,float(c)))+author.log_response(opt.baryons,float(qq),Enu-Ee) for c,w,qq in zip(cs,cws,q)]
  return math.log(author.a.K*T)+float(log_expit(y))+float(logsumexp(vals))
 scale=max(logfn(y) for y in [-30.,-10.,-3.,0.,3.,10.,30.]);value=0.;err=0.
 for lo,hi in zip(edges[:-1],edges[1:]):
  v,e=quad(lambda y:math.exp(logfn(y)-scale),lo,hi,epsabs=2e-13,epsrel=3e-11,limit=200);value+=v;err+=e
 return scale+math.log(value),err/value
other,err=adaptive_log_chi(64,60.);refined,err2=adaptive_log_chi(96,120.)
check('heldout angular/adaptive tail refinement',abs(math.expm1(other-refined)),3e-7)
check('heldout adaptive quadrature reported residual',err2,3e-8)
check('heldout author72 versus independent adaptive-cos',abs(math.expm1(fine[1]-refined)),3e-6)
check('heldout log KMS',abs(fine[0]-fine[1]+(Enu-opt.mu_neutrino)/T),3e-10)
heldout_record=dict(Y=Y,T=T,Enu=Enu,offset=delta,author_logs=fine[:2],independent_log_chi=refined,independent_quadrature_error=err2)
# Independent dense angular/electron cold-support grid, not author scalar
# optimized q formula. It tests onset signs on either side, not exact onset.
cold=[]
for Y in (.10,.105,.12):
 nn=.01*(1-Y);np_=.01*Y;pn=author.eos.HBARC*(3*math.pi**2*nn)**(1/3);pp=author.eos.HBARC*(3*math.pi**2*np_)**(1/3)
 en=pn*pn/(2*M);ep=pp*pp/(2*M);mue=math.hypot(pp,author.a.ME);delta=ep+mue-en
 B=(pn-pp)**2-pp*pp;A=M+pn-pp
 critical=max(0.,B/(A+math.sqrt(A*A-B)))
 for E in ((.95*critical,1.05*critical) if critical else (.03,.3)):
  ees=np.linspace(mue,mue+E,301)[:,None];cs=np.linspace(-1,1,401)[None,:]
  pe=np.sqrt((ees-author.a.ME)*(ees+author.a.ME));q=np.sqrt((E-pe)**2+2*E*pe*(1-cs));w=E-ees+delta
  emin=(M*w/q-q/2)**2/(2*M);interval=np.maximum(0.,en-np.maximum(emin,ep-w));support=float(interval.max())
  expected=E>critical;check(f'independent cold support Y{Y} E{E}',float((support>0)!=expected),.1)
  cold.append(dict(Y=Y,E=E,critical=critical,max_interval=support))
result=dict(status='PASS' if all(c['passed'] for c in checks) else 'FAIL',checks=checks,heldout=heldout_record,cold_support=cold,source_sha256=sourcehash,end_source_sha256=hashlib.sha256(sourced.read_bytes()).hexdigest(),runtime_seconds=time.monotonic()-start,limits='New selected point and independent density/cold-support checks; shared analytic response and weak prefactor; no new physical EOS or cold-extension qualification')
(OUT/'review_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result['status'],sum(c['passed'] for c in checks),'/',len(checks),result['runtime_seconds'])
for c in checks:
 if not c['passed']:print(c)
