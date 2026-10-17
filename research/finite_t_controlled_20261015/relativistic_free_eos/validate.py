"""Independent momentum quadrature, limits and first-derivative qualifications."""
from free_baryons import *
from scipy.integrate import quad
from scipy.special import kv
import time,json,hashlib,atexit
P=Path(__file__).resolve().parent;start=__import__('time').monotonic();checks=[];rows=[];failures=[]
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error) and error<tol)))
def save():(P/'results.json').write_text(json.dumps(dict(checks=checks,rows=rows,failures=failures,runtime_seconds=__import__('time').monotonic()-start),indent=2)+'\n')
atexit.register(save)
def independent(T,mu,m):
 nu=mu-m;end=max(nu,0)+70*T;ks=sorted(set([0.,end]+[nu+x*T for x in (-12,-4,0,4,12)if 0<nu+x*T<end]));ps=[math.sqrt(k*(k+2*m))for k in ks]
 def scalar(p,which):
  E=math.hypot(m,p);x=(E-m-nu)/T;f=float(expit(-x));w=f*float(expit(x));h=float(np.logaddexp(0.,-abs(x))+abs(x)*expit(-abs(x)))
  return p*p*[f,(E-m)*f,p*p/(3*E)*f,h,w,(E-m-nu)*w,(E-m-nu)**2*w][which]
 vals=[sum(quad(lambda p:scalar(p,i),a,b,epsabs=1e-20,epsrel=4e-11,limit=150)[0] for a,b in zip(ps[:-1],ps[1:]))*C for i in range(7)]
 return vals
for label,T,n,mass in [('dilute',5.,1e-8,MN),('warmFD',2.,.002,MP),('coldFD',.05,.08,MN),('warmrel',40.,.01,MP)]:
 s=state_density(T,n,mass);rows.append(dict(label=label,state=s.asdict()));gate(label+' forward density',abs(s.density/n-1),3e-10)
 iv=independent(T,s.chemical_potential,mass)
 for i,field in enumerate(('density','kinetic_energy_density','pressure','entropy_density')):gate(label+' independent momentum '+field,abs(iv[i]/getattr(s,field)-1),3e-9)
 independentCv=(iv[6]-iv[5]**2/iv[4])/T**2;gate(label+' independent momentum Cv',abs(independentCv/s.cv_fixed_density-1),3e-9)
 gate(label+' Euler thermodynamic identity',abs((s.energy_density+s.pressure-s.chemical_potential*s.density)/(T*s.entropy_density)-1),3e-9)
 for hfrac in ((1e-3,5e-4)if T<.1 else (1e-4,5e-5)):
  h=T*hfrac;plus=state_density(T+h,n,mass);minus=state_density(T-h,n,mass)
  cv=(plus.kinetic_energy_density-minus.kinetic_energy_density)/(2*h);gate(label+' Cv FD '+str(hfrac),abs(cv/s.cv_fixed_density-1),3e-6)
  mut=(plus.chemical_potential-minus.chemical_potential)/(2*h);gate(label+' muT FD '+str(hfrac),abs(mut-s.dmu_dtemperature_fixed_density)/max(abs(s.dmu_dtemperature_fixed_density),1e-8),3e-6)
 h=max(T*1e-4,1e-5);plus=state_mu(T,s.chemical_potential+h,mass);minus=state_mu(T,s.chemical_potential-h,mass)
 gate(label+' susceptibility FD',abs(((plus.density-minus.density)/(2*h))/s.dn_dmu_fixed_temperature-1),3e-6)
 # Forward-energy-cutoff refinement independently from momentum coordinates.
 tight=state_density(T,n,mass,tail=80.,rtol=5e-13)
 for field in ('chemical_potential','pressure','entropy_density','cv_fixed_density'):gate(label+' tail/refinement '+field,abs(getattr(tight,field)/getattr(s,field)-1),3e-9)
 save()
# Dilute exact relativistic Maxwell-Boltzmann shape and Bessel normalization.
T=25.;mu=MP-18*T;s=state_mu(T,mu,MP);z=MP/T
nMB=MP*MP*T*kv(2,z)*math.exp(mu/T)/(math.pi**2*HBARC**3);meanE=MP*kv(1,z)/kv(2,z)+3*T
gate('dilute Bessel n',abs(s.density/nMB-1),2e-8);gate('dilute Bessel meanE',abs((s.energy_density/s.density)/meanE-1),2e-8);gate('dilute classical pressure',abs(s.pressure/(s.density*T)-1),2e-8)
# Cold relativistic analytic integrals and Sommerfeld Cv (fixed density).
n=.08;pf=HBARC*(3*math.pi**2*n)**(1/3);ef=math.hypot(MN,pf);z=math.asinh(pf/MN);E0=(pf*ef*(2*pf*pf+MN*MN)-MN**4*z)/(8*math.pi**2*HBARC**3);P0=(pf*ef*(2*pf*pf-3*MN*MN)+3*MN**4*z)/(24*math.pi**2*HBARC**3);D=pf*ef/(math.pi**2*HBARC**3)
for T in (.05,.025,.0125):
 s=state_density(T,n);gate('cold energy T'+str(T),abs(s.energy_density/E0-1),3e-7);gate('cold pressure T'+str(T),abs(s.pressure/P0-1),5e-5);gate('Sommerfeld Cv T'+str(T),abs(s.cv_fixed_density/(math.pi**2/3*D*T)-1),3e-5)
# Controlled NR comparison uses same physical mass; no mass/Q substitution.
sys.path.insert(0,str(ROOT/'nr_baryon_eos'));import free_nucleons as nr
for T,n in [(.001,1e-9),(.01,1e-8)]:
 a=state_density(T,n,MP);b=nr.state_density(T,n,offset=MP,mass=MP)
 for name,x,y in [('kinetic',a.kinetic_energy_density,b.kinetic_energy_density),('pressure',a.pressure,b.pressure),('entropy',a.entropy_density,b.entropy_density),('Cv',a.cv_fixed_density,b.cv_fixed_density)]:gate('NR limit '+str(T)+name,abs(x/y-1),7e-5)
# Independent composition, explicit masses and pair lepton zero boundary.
for T,n,Ye,Ymu in [(2.,.001,.15,.03),(10.,.01,.2,.05),(3.,.002,.2,0.),(3.,.002,.25,-.02)]:
 m=matter_state(T,n,Ye,Ymu);rows.append(dict(label='matter',state=m.asdict()));d=matter_derivatives(m)
 gate('matter baryon conservation '+str((T,Ye,Ymu)),abs((m.neutron.density+m.proton.density)/n-1),3e-10)
 gate('matter independent charge '+str((T,Ye,Ymu)),abs((m.electron.net_density+m.muon.net_density-m.proton.density)/n),3e-10)
 gate('physical neutron mass',abs(m.neutron.mass-MN),1e-12);gate('physical proton mass',abs(m.proton.mass-MP),1e-12);gate('explicit muon mass',abs(m.muon.mass-leptons.MUON_MASS),1e-12)
 for key,h in [('T',T*1e-4),('nB',n*1e-4),('Ye',1e-5),('Ymu',1e-5)]:
  args=[T,n,Ye,Ymu];ix={'T':0,'nB':1,'Ye':2,'Ymu':3}[key];ap=args.copy();am=args.copy();ap[ix]+=h;am[ix]-=h
  up=matter_state(*ap).energy_density/ap[1];um=matter_state(*am).energy_density/am[1];measured=(up-um)/(2*h);anal=d['u_'+key]
  gate('matter specific energy derivative '+str((T,Ye,Ymu))+key,abs(measured-anal)/max(abs(anal),1e-8),3e-6)
 recovered=temperature_from_energy(m.energy_density,n,Ye,Ymu,T*.9);gate('matter energy inversion '+str((T,Ye,Ymu)),abs(recovered/T-1),2e-8)
 save()
# Explicit invalid/domain counterexamples: no species floors, high-antibaryon rejection.
for label,call in [('zero baryon',lambda:state_density(2.,0)),('pure neutron zero proton',lambda:matter_state(2.,.01,0.,0.)),('pure proton zero neutron',lambda:matter_state(2.,.01,1.,0.)),('nan density',lambda:state_density(2.,math.nan)),('negative T',lambda:state_density(-1.,.01)),('antibaryon dense',lambda:state_mu(200.,0.,MN)),('original dilute antiparticle exclusion',lambda:state_mu(37.,MP-18*37.,MP)),('tail nonfinite',lambda:state_density(2.,.01,tail=math.nan))]:
 try:call();rejected=False
 except ValueError:rejected=True
 gate('reject '+label,float(not rejected),.1)
# Finite logarithmic antipop diagnostics distinguish underflow from physical zero.
s=state_density(2.,.002);gate('underflow diagnostic finite log',float(not math.isfinite(s.log_antiparticle_density)),.1);gate('underflow diagnostic ordinary zero flagged',float(s.antiparticle_density!=0 or s.antiparticle_density_representable),.1)
save();print(sum(c['passed']for c in checks),'/',len(checks),'runtime',__import__('time').monotonic()-start)
for c in checks:
 if not c['passed']:print('FAIL',c)
