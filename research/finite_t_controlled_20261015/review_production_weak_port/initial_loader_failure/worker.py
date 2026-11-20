"""Pure selected explicit-chemical reference; never imports productioncandidate."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util,sys,json,time,math,hashlib
import numpy as np
from scipy.integrate import quad
from scipy.special import expit
P=Path(__file__).resolve().parent;R=P.parent;sys.dont_write_bytecode=True
sp=importlib.util.spec_from_file_location('weak_review_unique_oracle_loader',P/'oracles.py');O=importlib.util.module_from_spec(sp);sys.modules[sp.name]=O;sp.loader.exec_module(O)
sp=importlib.util.spec_from_file_location('weak_review_unique_baryon_momentum',R/'review_production_baryon_port/oracle.py');D=importlib.util.module_from_spec(sp);sys.modules[sp.name]=D;sp.loader.exec_module(D)
L=O.LOCK;i=int(sys.argv[1]);profile=sys.argv[2];assert i in[0,1]and profile in L['profiles'];q=L['baths'][i];cp=L['profiles'][profile];T=q['T'];mun,mup,mue=[q[k]for k in['mu_n','mu_p','mu_e']];eq=mun-mup-mue;b=SimpleNamespace(T=T,mu_n=mun,mu_p=mup,mu_e=mue,equilibrium_mu_nue=-eq);checks=[];raw={};start=time.monotonic();stage='density';forward={}
def ck(name,error,gate):checks.append(dict(name=name,error=float(error),gate=gate,passed=bool(abs(error)<=gate)))
def enc(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 raise TypeError(str(type(x)))
def save():
 (P/f'pure_{i}_{profile}.json').write_text(json.dumps(dict(stage=stage,bath=q,profile=profile,controls=cp,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],checks=checks,forward=forward,channels=raw,seconds=time.monotonic()-start),default=enc,indent=2,allow_nan=False)+'\n')
def nofail():
 save()
 if any(not c['passed']for c in checks):raise ArithmeticError('Initialforward/phasefailurepreserved')
# FIRST actual given-muquantumFD densities, independentofeventmomentumrules.
for name,m,mu in [('neutron',O.MN,mun),('proton',O.MP,mup)]:
 z=D.momentum(T,mu,m);bz=D.fugacity(T,mu,m);forward[name]=z
 for field,key in [('number','number'),('energy','energy'),('pressure','pressure')]:ck('givenmuoriginalmomentum/Bessel '+name+field,z[field]/bz[key]-1,3e-9)
def lepton(mu):return sum(quad(lambda y:y*y*expit((mu-math.hypot(T*y,O.ME))/T),a,c,epsabs=1e-18,epsrel=2e-12)[0]for a,c in zip([0,1,3,8,20,60],[1,3,8,20,60,120]))*T**3/(math.pi**2*O.HC**3)
forward['electron_minus_density']=lepton(mue);forward['positron_density']=lepton(-mue);forward['net_electron_density']=forward['electron_minus_density']-forward['positron_density'];forward['nonneutral_charge_residual']=forward['proton']['number']-forward['net_electron_density'];forward['baryon_density']=forward['neutron']['number']+forward['proton']['number']
ck('actualbothpairpopulationspositive',0 if min(forward['electron_minus_density'],forward['positron_density'])>0 else 1,0);ck('explicitNONneutraldeclared',0 if 'NONNEUTRAL'in q['label']and abs(forward['nonneutral_charge_residual'])>1e-10 else 1,0);nofail()
# EC independentphase/directtrace, channelradiationselects ONLYnue.
stage='EC';p=cp['EC'];rad={n:None if r is None else r['mu_nue']for n,r in q['radiation'].items()};raw['EC']=O.EC.integrate(b,*p,radiation_mus=rad);ck('EC actualgivenmu protondensity',raw['EC']['density']/forward['proton']['number']-1,3e-9);ck('ECeventonshell',raw['EC']['maxfour']/O.MN,2e-10);nofail()
# PC independentreviewerpolynomial andpmap, directentropyF/Bratio.
stage='PC';p=cp['PC'];rad=[None if r is None else r['mu_antinue']for r in q['radiation'].values()];raw['PC']=O.PC.integral(vars(b),order_n=p[0],order_e=p[1],order_angle=p[2],tail=p[3],budget=105,radiation_mus=rad);ck('PC actualgivenmu neutrondensity',raw['PC']['neutron_density']/forward['neutron']['number']-1,3e-9);ck('PCeventonshell',raw['PC']['max_shell'],2e-10);nofail()
# Beta independentflatDalitz/sourcepolynomial/directparentmomentum.
stage='NEUTRON_DECAY';p=cp['NEUTRON_DECAY'];rad={n:None if r is None else r['mu_antinue']for n,r in q['radiation'].items()};raw['NEUTRON_DECAY']=O.BETA.integrate(b,0.,order=p[0],radial=p[1],orient=p[2],phi=p[3],budget=105,radiation_mus=rad,parent_tail=p[4]);ck('beta actualgivenmu neutrondensity',raw['NEUTRON_DECAY']['density']/forward['neutron']['number']-1,3e-9);ck('betaeventonshell',raw['NEUTRON_DECAY']['max_fourmomentum_residual']/O.MN,2e-10)
# Propernumber/DopplerenergyanchorsareUNBLOCKEDlimitingdiagnostics only.
scl=math.sqrt(2*O.MN*T);proper=quad(lambda y:y*y*expit((mun-math.hypot(O.MN,scl*y))/T)*O.MN/math.hypot(O.MN,scl*y),0,math.sqrt(130),epsabs=1e-20,epsrel=3e-12)[0]*scl**3/(math.pi**2*O.HC**3);z=raw['NEUTRON_DECAY'];un=z['scenarios']['unblocked']['F'];ck('unblockedproperbetanumber',un[0]/(proper*z['rest_rate'])-1,3e-8)
for idx,key in[(1,'neutrino_energy'),(2,'electron_energy'),(3,'proton_energy')]:ck('unblockedDoppler '+key,un[idx]/(forward['neutron']['number']*z['rest_moments'][key])-1,3e-8)
for rel,h in L['sourcepins'].items():
 if O.sha(R/rel)!=h:raise RuntimeError('Scientificsourcechanged '+rel)
stage='COMPLETE';nofail();print('PURE',i,profile,'PASS',sum(c['passed']for c in checks),'FAIL',sum(not c['passed']for c in checks),'seconds',time.monotonic()-start)
