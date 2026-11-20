"""Separate independent recheck: old failures, stable MB moments and new ideal-gas states."""
from pathlib import Path
import sys,math,json,hashlib,importlib.util,ast,dataclasses
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import expit,kve
from scipy.integrate import quad
P=Path(__file__).resolve().parent;ROOT=P.parent;A=ROOT/'production_lepton_port';sys.dont_write_bytecode=True;sys.path.insert(0,str(A/'test_mirror/src'))
from nmbackend.reference import finite_t_leptons as L
from nmbackend.constants import HBARC,M_E,M_MU
import jax
from nmbackend.eos.leptons import lepton_sector
jax.config.update('jax_enable_x64',True)
checks=[];rows=[];before={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in A.rglob('*')if p.is_file()}
def ck(n,e,t=3e-8):checks.append(dict(name=n,error=float(e),gate=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
def rel(a,b):return a/b-1 if b else a
def compare(label,s,ref,t=3e-8):
 for k,v in ref.items():ck(label+'/'+k,rel(getattr(s,k),v),t if v else 0.)
sp=importlib.util.spec_from_file_location('nmbackend.reference.review_preserved_initial_leptons',A/'pre_quadrature_repair/staged/src/nmbackend/reference/finite_t_leptons.py');old=importlib.util.module_from_spec(sp);sys.modules[sp.name]=old;sp.loader.exec_module(old)
ck('repaired source SHA',0 if hashlib.sha256(Path(L.__file__).read_bytes()).hexdigest()=='e562135234bcbd9c27e56e5b622253d5a3d2420c52c0b3fc67f5f383918e5a76'else 1,0)
ck('candidate manifest SHA',0 if before['production_lepton_port/CANDIDATE_MANIFEST.json']=='599bc681945ff344d987f77a557565337be44ce17509fabbca2d36bd1c47de7c'else 1,0)
# Independent all-field dilute oracle: Bessel absolute population and stable
# original-momentum centered kinetic variance, avoiding tiny squared moments.
def mb(T,mu,m,N=96):
 z=m/T;alpha=mu/T;base=m*m*T/(math.pi**2*HBARC**3)*kve(2,z)*math.exp(-z);na=base*math.exp(alpha);nb=base*math.exp(-alpha);total=na+nb;net=2*base*math.sinh(alpha)
 x,w=leggauss(N);upper=math.sqrt(200*(1+50/z));x=(x+1)*upper/2;w=w*upper/2;k=x*x/(np.sqrt(1+x*x/z)+1);wt=w*x*x*np.exp(-k);norm=sum(wt);mean_k=sum(wt*k)/norm;variance=sum(wt*(k-mean_k)**2)/norm;mean=m+T*mean_k
 en=total*mean;pressure=total*T;chi=total/T;nt=(net*mean/T-alpha*total)/T;cv=total*variance+(4*na*(nb/total))*(mean/T)**2
 return dict(net_density=net,particle_density=na,antiparticle_density=nb,energy_density=en,pressure=pressure,entropy_density=en/T+total-alpha*net,free_energy_density=mu*net-pressure,cv_fixed_net_density=cv,dn_dmu_fixed_temperature=chi,dn_dtemperature_fixed_mu=nt,dmu_dtemperature_fixed_net_density=-nt/chi)
for z in [40.,100.,300.,600.]:
 T=M_E/z;ref=mb(T,0.,M_E);previous=old.state_mu(T,0.);new=L.state_mu(T,0.);old_error=previous.particle_density/ref['particle_density']-1
 ck('exact original failure remains reproduced '+str(z),0 if abs(old_error)>3e-8 else 1,0);compare('repaired old failure allfields '+str(z),new,ref);rows.append(dict(z=z,old_error=old_error,new_error=new.particle_density/ref['particle_density']-1))
 # New signed near-dilute heldouts beyond the author's +/-.4 choices.
 for alpha in [-.53,.53]:
  s=L.state_mu(T,alpha*T);ref=mb(T,alpha*T,M_E);fine=mb(T,alpha*T,M_E,144);compare('new dilute '+str((z,alpha)),s,ref)
  for k,v in ref.items():ck('MB momentum refinement '+str((z,alpha,k)),rel(v,fine[k]),3e-10 if v else 0)
  f=L.state_mu(T,alpha*T,tail=90.,rtol=4e-13)
  for k in ref:ck('dilute actual source refine '+str((z,alpha,k)),rel(getattr(s,k),getattr(f,k)),3e-10 if getattr(f,k)else 0)
# Reuse only the independent oracle FUNCTION source, never import its driver.
oldtext=(ROOT/'review_production_lepton_port/probe.py').read_text();node=next(n for n in ast.parse(oldtext).body if isinstance(n,ast.FunctionDef)and n.name=='oracle');exec(ast.get_source_segment(oldtext,node),globals())
for T,mu,m in[(.63,1.19,M_E),(4.73,-.83,M_E),(18.23,-98.7,M_MU),(.217,147.3,M_MU)]:
 s=L.state_mu(T,mu,m);a=oracle(T,mu,m,72);b=oracle(T,mu,m,112);b['free_energy_density']=mu*b['net_density']-b['pressure'];compare('NEW warm/degenerate '+str((T,mu,m)),s,b)
 for k,v in a.items():ck('NEW oracle refinement '+str((T,mu,k)),rel(v,b[k]),3e-8 if b[k]else 0)
 # Direct rest-inclusive cold subtraction is well conditioned in warm cases.
 if T>.5:
  pf=HBARC*(3*math.pi**2*abs(s.net_density))**(1/3);mc=math.hypot(pf,m);e0=(pf*mc*(2*pf*pf+m*m)-m**4*math.asinh(pf/m))/(8*math.pi**2*HBARC**3)
  ck('warm independent thermal increment '+str((T,mu)),L.thermal_energy_density(s)/(b['energy_density']-e0)-1,3e-8)
 rows.append(dict(heldout=[T,mu,m],state=s.asdict()))
# NEW dimensionful massless scalings, analytic formulas in alpha avoid scale loss.
for T in [2e-41,7e-7,3e7]:
 for alpha in [0.,-.61,.61]:
  mu=alpha*T;h3=HBARC**3;nd=alpha**3/(3*math.pi**2)+alpha/3;pd=alpha**4/(12*math.pi**2)+alpha*alpha/6+7*math.pi**2/180;sd=alpha*alpha/3+7*math.pi**2/45;ch=alpha*alpha/math.pi**2+1/3;nt=2*alpha/3
  n=T**3/h3*nd;pressure=T**4/h3*pd;cv=T**3/h3*(alpha*alpha/3+7*math.pi**2/15-nt*nt/ch);particle=T**3/(math.pi**2*h3)*quad(lambda x:x*x*expit(alpha-x),0,np.inf,epsabs=1e-12,epsrel=2e-13)[0];anti=T**3/(math.pi**2*h3)*quad(lambda x:x*x*expit(-alpha-x),0,np.inf,epsabs=1e-12,epsrel=2e-13)[0]
  ref=dict(net_density=n,particle_density=particle,antiparticle_density=anti,energy_density=3*pressure,pressure=pressure,entropy_density=T**3/h3*sd,free_energy_density=mu*n-pressure,cv_fixed_net_density=cv,dn_dmu_fixed_temperature=T*T/h3*ch,dn_dtemperature_fixed_mu=T*T/h3*nt,dmu_dtemperature_fixed_net_density=-nt/ch)
  s=L.state_mu(T,mu,0.);compare('NEW massless scale '+str((T,alpha)),s,ref,3e-10);cold=T**4/h3*(3*math.pi**2*abs(nd))**(4/3)/(4*math.pi**2);ck('NEW massless thermal increment '+str((T,alpha)),L.thermal_energy_density(s)/(3*pressure-cold)-1,3e-10)
# Response FD only after the independent heldout forward state passes.
for m,T,mu in[(M_E,4.73,-.83),(M_MU,18.23,-98.7)]:
 s=L.state_mu(T,mu,m);h=2e-4*T;lo=L.state_density(T-h,s.net_density,m);hi=L.state_density(T+h,s.net_density,m);ck('NEW fixed-net Cv '+str(m),(hi.energy_density-lo.energy_density)/(2*h)/s.cv_fixed_net_density-1,1e-6);ck('NEW fixed-net muT '+str(m),(hi.chemical_potential-lo.chemical_potential)/(2*h)/s.dmu_dtemperature_fixed_net_density-1,1e-6)
for m,T in[(M_E,.91),(M_MU,19.1)]:
 pair=L.state_mu(T,0.,m)
 for n in [-4.7e-29,4.7e-29]:
  s=L.state_density(T,n,m);ck('NEW massive signed tiny root '+str((m,n)),s.net_density/n-1,1e-9);ck('NEW tiny susceptibility '+str((m,n)),s.chemical_potential/(n/pair.dn_dmu_fixed_temperature)-1,1e-9)
s=L.state_mu(.0009,173.2,M_MU);pf=math.sqrt(173.2**2-M_MU**2);n0=pf**3/(3*math.pi**2*HBARC**3);cv=n0*.0009*math.pi**2*173.2/pf**2;ck('NEW Sommerfeld Cv',s.cv_fixed_net_density/cv-1,3e-8);ck('NEW cold thermal increment',L.thermal_energy_density(s)/(.0009*cv/2)-1,3e-8)
out=jax.jit(lambda x:lepton_sector(x,muons=True))(.0173);mu=float(out[0])*HBARC;e=L.state_mu(.0009,mu,M_E);m=L.state_mu(.0009,mu,M_MU);ck('actual cold number units',(e.net_density+m.net_density)/float(out[1]+out[2])-1,1e-7);ck('actual cold energy units',(e.energy_density+m.energy_density)/(float(out[3])*HBARC)-1,1e-7);hess=jax.grad(jax.grad(lambda x:lepton_sector(x,muons=True)[3]))(.0173);ck('actual cold Hessian finite',0 if math.isfinite(float(hess))and hess>0 else 1,0)
# Explicit guards and ordinary underflow remain distinct.
try:L.state_mu(.0008,0.,M_MU);failed=False
except ArithmeticError:failed=True
ck('unresolved susceptibility rejects',0 if failed else 1,0);s=L.state_mu(.03,180.,M_MU);ck('ordinary antiparticle underflow accepted with positive susceptibility',0 if s.antiparticle_density==0 and s.dn_dmu_fixed_temperature>0 else 1,0)
for bad in [True,np.bool_(False),float('nan'),float('inf'),np.array(1.),'1']:
 for which in ['T','mu','mass','rtol','tail','net']:
  args=[1.,0.,M_E];kw={}
  if which in ['T','mu','mass']:args[['T','mu','mass'].index(which)]=bad
  elif which in ['rtol','tail']:kw[which]=bad
  try:
   if which=='net':L.state_density(1.,bad)
   else:L.state_mu(*args,**kw)
   ok=False
  except ValueError:ok=True
  ck('finite-real nonBool '+str((which,repr(bad))),0 if ok else 1,0)
# Snapshot and original independent failures must remain byte-for-byte frozen.
for n,h in json.loads((A/'INITIAL_INDEPENDENT_REVIEW_PIN.json').read_text()).items():ck('initial review immutable '+n,0 if hashlib.sha256((ROOT/'review_production_lepton_port'/n).read_bytes()).hexdigest()==h else 1,0)
manifest=json.loads((A/'CANDIDATE_MANIFEST.json').read_text());ck('exact five staged paths',0 if set(manifest['files'])=={'README.md','tests/test_finite_t_leptons.py','docs/finite_temperature_leptons.md','examples/finite_t_leptons.py','src/nmbackend/reference/finite_t_leptons.py'}else 1,0)
for n,h in manifest['files'].items():ck('staged SHA '+n,0 if hashlib.sha256((A/'staged'/n).read_bytes()).hexdigest()==h else 1,0)
for n in manifest['files']:
 if n.endswith('.py'):ast.parse((A/'staged'/n).read_text(),feature_version=(3,9));ck('Python3.9 static grammar '+n,0,0)
after={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in A.rglob('*')if p.is_file()};ck('all author bytes unchanged',int(before!=after),0)
out=dict(checks=checks,passed=sum(x['passed']for x in checks),failed=[x for x in checks if not x['passed']],rows=rows,before=before,after=after,environment=dict(python=sys.version,numpy=np.__version__,jax=jax.__version__),origins={k:v.__file__ for k,v in sys.modules.copy().items()if k.startswith('nmbackend')and hasattr(v,'__file__')});(P/'results.json').write_text(json.dumps(out,indent=2)+'\n');print(out['passed'],out['failed'])
