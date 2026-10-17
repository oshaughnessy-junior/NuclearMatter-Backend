"""Review-only Bessel baryons and original-momentum neutral lepton EOS."""
from pathlib import Path
import sys,math,json,hashlib,importlib.util
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import expit,kve
from scipy.integrate import quad_vec
from scipy.optimize import brentq
P=Path(__file__).resolve().parent;ROOT=P.parent;A=ROOT/'fit_balance_assessment';sys.dont_write_bytecode=True
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
C=load('review_fit_balance_direct_ec','hot_electron_capture_moments/electron_capture.py')
BETA=load('review_fit_balance_direct_beta','hot_beta_moments/hot_moments.py')
PC=load('review_fit_balance_direct_pc','review_pc_moments/reference.py')
F=load('review_fit_balance_positive_API','free_capture_moment_fit/approximation.py')
HC=C.HC
AUTHOR=json.loads((A/'results.json').read_text());LOCK=json.loads((A/'LOCK.json').read_text())
def inventory():
 dirs=['fit_balance_assessment','free_capture_moment_fit','review_free_capture_moment_fit_repair','transparent_balance_grid','review_transparent_balance_grid']
 out={str(p.relative_to(ROOT)):sha(p)for d in dirs for p in (ROOT/d).rglob('*')if p.is_file()and '__pycache__'not in str(p)}
 out.update({r:sha(ROOT/r)for r in AUTHOR['pins']})
 return out
def verify():
 assert sha(A/'assess.py')=='2d591cfe2c0687d068c14f3f38c9ca19144688d1b6c4a2b391f32fc824639b2d'
 assert sha(A/'MANIFEST.json')=='196f665fd97612247a79d68d9d95193ac90086e1cbf226de7c0e67ffdc6a6ac6'
 for r,h in json.loads((A/'MANIFEST.json').read_text())['files'].items():assert sha(A/r)==h,r
 for r,h in AUTHOR['pins'].items():assert sha(ROOT/r)==h,r
 acceptance=json.loads((ROOT/'transparent_balance_grid/EC_REVIEW_ACCEPTANCE.json').read_text())
 assert acceptance['accepted']is True and sha(ROOT/acceptance['review_path'])==acceptance['review_sha256']
 C.verify_sources();BETA.verify_sources() if hasattr(BETA,'verify_sources')else None
verify()
def bessel_density(T,eta,m):
 k=np.arange(1,33);return m*m*T/(math.pi**2*HC**3)*np.sum((-1.)**(k+1)*np.exp(k*eta)*kve(2,k*m/T)/k)
def baryon(T,n,m):
 eta=brentq(lambda z:bessel_density(T,z,m)/n-1,-100.,-.001,xtol=2e-13,rtol=1e-14);mu=m+T*eta
 scale=math.sqrt(m*T)
 def fun(q):
  p=scale*q;E=math.hypot(p,m);kin=p*p/(E+m);f=expit(eta-kin/T);w=f*(1-f)
  return q*q*np.array([f,E*f,w,E*w])
 vals=scale**3/(math.pi**2*HC**3)*quad_vec(fun,0.,30.,epsabs=0.,epsrel=3e-12)[0]
 return dict(mu=mu,eta=eta,density=vals[0],energy=vals[1],epsilon=vals[3]/vals[2],chi=vals[2]/T)
def electron(T,mu,order=80):
 x,w=leggauss(order);data=np.zeros(8)
 for lo,hi in zip([0.,1.,4.,10.,30.],[1.,4.,10.,30.,90.]):
  p=T*(lo+(x+1)*(hi-lo)/2);E=np.hypot(p,C.ME);a=expit((mu-E)/T);b=expit((-mu-E)/T);wa=a*(1-a);wb=b*(1-b)
  weight=w*(hi-lo)*T/2*p*p/(math.pi**2*HC**3)
  data+=np.array([weight@(a-b),weight@(E*(a+b)),weight@(wa+wb),weight@(E*(wa-wb)),weight@a,weight@b,weight@wa,weight@wb])
 return dict(density=data[0],energy=data[1],chi=data[2]/T,epsilon=data[3]/data[2],nminus=data[4],nplus=data[5],minus_fraction=data[6]/data[2],plus_fraction=data[7]/data[2])
def eos(T,n,y):
 neutron=baryon(T,n*(1-y),C.MN);proton=baryon(T,n*y,C.MP)
 mu=brentq(lambda z:electron(T,z)['density']/(n*y)-1,0.,100.,xtol=2e-13,rtol=1e-14);lepton=electron(T,mu)
 return dict(T=T,mu_n=neutron['mu'],mu_p=proton['mu'],mu_e=mu),dict(neutron=neutron,proton=proton,electron=lepton,energy=neutron['energy']+proton['energy']+lepton['energy'],uY=proton['epsilon']-neutron['epsilon']+lepton['epsilon'])
