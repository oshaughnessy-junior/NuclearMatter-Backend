"""Independent EOS/root and source ledgers; no frozen driver execution."""
from pathlib import Path
import importlib.util,sys,json,math,hashlib,time
import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar,brentq
from scipy.special import expit
P=Path(__file__).resolve().parent;ROOT=P.parent;sys.dont_write_bytecode=True

def load(name,path):
 s=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
M=load('independent_balance_review_author_functions','transparent_capture_balance/balance.py')
checks=[];rows=[];start=time.monotonic()
def ck(name,error,gate):
 checks.append(dict(name=name,error=float(error),gate=gate,passed=bool(math.isfinite(error)and abs(error)<=gate)))
def save():
 (P/'raw.json').write_text(json.dumps(dict(checks=checks,rows=rows,seconds=time.monotonic()-start),indent=2)+'\n')
def snap():
 paths=[p for p in (ROOT/'transparent_capture_balance').iterdir()if p.is_file()]
 paths += [ROOT/p for p in M.PINS]+[ROOT/p for p in json.loads((ROOT/'transparent_capture_balance/UPSTREAM.json').read_text())]
 paths += [ROOT/'review_hot_electron_capture/REVIEW.md',ROOT/'review_hot_electron_capture/probe.py']
 return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths}
before=snap();(P/'before.json').write_text(json.dumps(before,indent=2)+'\n')
ck('balance frozen SHA',int(before['transparent_capture_balance/balance.py']!='bbcf98a41d1c5eb6907eb0611da397c175e55681435193bf3af577b7bb8419c5'),0)
ck('checkpoint frozen SHA',int(before['transparent_capture_balance/CHECKPOINT.json']!='7e8fb5918c41b9241f5f7b4f0c7eaf0bd7cc601203bba5947b46b16a06f3db80'),0)
ck('EC independent acceptance pinned before scientific join',int(before['review_hot_electron_capture/REVIEW.md']!='2113dca85edaba56f67af36fc745a41d5d1e4e93db3eca6117ccac2eeca97aaf'),0)
M.require_acceptance()
T=2.;NB=1e-7;HC=197.3269804;MN=939.56542052;MP=938.27208816;ME=.51099895069
# Infinite momentum interval, scalar QUADPACK vs frozen finite kinetic-energy quad_vec.
def density(mu,m,anti=False):
 scale=T if m<1 else math.sqrt(2*m*T)
 def f(x):
  p=x*scale;E=math.hypot(m,p);v=expit((mu-E)/T)
  if anti:v-=expit((-mu-E)/T)
  return x*x*v
 return scale**3/(math.pi**2*HC**3)*quad(f,0,np.inf,epsabs=1e-18,epsrel=3e-12,limit=160)[0]
def eos(y):
 un=brentq(lambda v:density(MN+v,MN)/(NB*(1-y))-1,-60,0,xtol=1e-12)
 up=brentq(lambda v:density(MP+v,MP)/(NB*y)-1,-60,0,xtol=1e-12)
 ue=brentq(lambda v:density(v,ME,True)/(NB*y)-1,0,3,xtol=1e-13)
 return dict(T=T,mu_n=MN+un,mu_p=MP+up,mu_e=ue)

def source(y,fine=False):
 # Own EOS constructed for every call, never cached sources.
 b=eos(float(y));bath=M.EC.Bath(**b)
 ec=M.EC.integrate(bath,proton_order=30 if fine else 24,electron_order=50 if fine else 40,initial_order=12 if fine else 10,outgoing_order=12 if fine else 10,phi_order=12 if fine else 10,tail=70. if fine else 55.,scenarios=[dict(name='empty',mode='empty')])
 pc=M.PC.integral(b,order_n=34 if fine else 30,order_e=54 if fine else 48,order_angle=12 if fine else 10,tail=70. if fine else 55.)
 beta=M.BETA.integrate(M.BETA.K.Bath(**b),rest_order=26 if fine else 20,parent_order=30 if fine else 26,orient_order=10 if fine else 8,phi_order=16 if fine else 12,coordinate='electron',parent_coordinate='direct_p',tail=90. if fine else 75.,scenarios=[dict(name='empty',mode='empty')])
 channels=dict(EC=ec['scenarios']['empty']['net'],PC=dict(zip(M.PC.NAMES,np.asarray(pc['sums'])[0,0].tolist())),beta=beta['scenarios']['empty']['net'])
 Rb,Rp,Re=[channels[c]['number']for c in ['beta','PC','EC']];R=Rb+Rp-Re;Q=sum(c['neutrino_energy']for c in channels.values())
 # Rows (n,p,e-,e+,nu,anti). Literal channel stoichiometry, independent matrix ledger.
 S=np.array([[-1,-1,1],[1,1,-1],[1,0,-1],[0,-1,0],[0,0,1],[1,1,0]],float)
 ds=S@np.array([Rb,Rp,Re]);gross=Rb+Rp+Re
 for name,v in dict(baryon=ds[0]+ds[1],charge=ds[1]-ds[2]+ds[3],Le=ds[2]-ds[3]+ds[4]-ds[5]).items():ck(f'{y} {fine} {name}',v/gross,1e-14)
 energy=channels['EC']['neutron_energy']-channels['EC']['proton_energy']-channels['EC']['electron_energy']+sum(channels[c]['proton_energy']-channels[c]['neutron_energy']-channels[c]['positron_energy'if c=='PC'else'electron_energy']*(1 if c=='PC'else -1)for c in ['PC','beta'])
 ck(f'{y} {fine} restinclusive species energy',(energy+Q)/Q,3e-10)
 ck(f'{y} {fine} direct material loss',(sum(c['material_energy_loss']for c in channels.values())-Q)/Q,3e-10)
 for c,actual,target in [('EC',ec['proton_density'],NB*y),('PC',pc['neutron_density'],NB*(1-y)),('beta',beta['parent_density'],NB*(1-y))]:ck(f'{y} {fine} incoming FD density {c}',actual/target-1,3e-8)
 row=dict(Ye=float(y),fine=fine,bath=b,channels=channels,R=R,dYe_dt=R/NB,Q=Q,affinity=b['mu_n']-b['mu_p']-b['mu_e'],species_number_sources=ds.tolist(),raw=dict(EC=ec,PC=pc,beta=beta))
 rows.append(row);save();print('source',y,fine,R,Q,flush=True);return row

if __name__=='__main__':
 # Each complete experiment bounded below120s; source primitives each already <=110s.
 for y in [.3,.58,.6145204990481219]:
  b=eos(y);ba,st=M.material(y)
  for k in ['mu_n','mu_p','mu_e']:ck('independent EOS '+str(y)+' '+k,b[k]-getattr(ba,k),3e-10)
  ck('independent net charge '+str(y),density(b['mu_e'],ME,True)/(NB*y)-1,1e-10)
 chemical=root_scalar(lambda y:(lambda b:b['mu_n']-b['mu_p']-b['mu_e'])(eos(y)),bracket=[.5,.75],method='toms748',xtol=1e-12).root
 ck('independent chemical root',chemical-.6145204990481219,2e-10)
 lo=source(.54);hi=source(.58);ck('fresh root bracket signs',0 if lo['R']>0>hi['R']else 1,0)
 rt=root_scalar(lambda y:source(y)['R']/NB,bracket=[.54,.58],method='toms748',xtol=1e-10).root
 root=source(rt,True);chemical_source=source(chemical,True)
 ck('independent kinetic root',rt-.5560448218021664,2e-7)
 ck('root net absolute dYe_dt',root['dYe_dt'],3e-6)
 ck('nonzero root affinity',0 if root['affinity']>.5 else 1,0)
 ck('chemical root fresh actual drift',chemical_source['dYe_dt']/(-1.77646339955)-1,3e-6)
 ck('root escaped energy',root['Q']/1.33763294976682e-5-1,3e-6)
 coarse=min((r for r in rows if not r['fine']),key=lambda r:abs(r['Ye']-rt))
 # source at identical composition; avoid confusing quadrature with composition shifts.
 exactcoarse=source(rt)
 for c in root['channels']:
  for k,v in root['channels'][c].items():ck('sameYe independent rules '+c+' '+k,exactcoarse['channels'][c][k]/v-1,3e-6)
 after=snap();ck('frozen bytes unchanged',0 if before==after else 1,0)
 (P/'summary.json').write_text(json.dumps(dict(kinetic=rt,chemical=chemical,balanced=root,chemical_source=chemical_source,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],seconds=time.monotonic()-start),indent=2)+'\n');save()
 print('DONE',len(checks),'failures',sum(not c['passed']for c in checks),'seconds',time.monotonic()-start,flush=True)
