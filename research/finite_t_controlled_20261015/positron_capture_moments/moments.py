"""Physical total n+e+ -> p+antinue events, direct reciprocal FD products.

Research prototype; common-T isotropic FD radiation or empty radiation only.
"""
from pathlib import Path
import sys,math,time
import numpy as np
from scipy.special import log_expit,logsumexp
from numpy.polynomial.legendre import leggauss
R=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True
sys.path.insert(0,str(R/'positron_capture'));import pc_kernel as PC
sys.path.insert(0,str(R/'relativistic_capture_oracle'));import events as E
HC=E.HC;HBAR=E.HBAR;MN=PC.crossed.MN;MP=PC.crossed.MP;ME=PC.ME;MW=PC.crossed.MW
NAMES=('number','neutrino_energy','positron_energy','proton_energy','neutron_energy','material_energy_loss')

def grid(order,scale,upper):
 # kinetic E=scale*y²; exact Jacobian and no physical support modification.
 x,w=leggauss(order);y=(x+1)*math.sqrt(upper/scale)/2
 return scale*y*y,w*math.sqrt(upper/scale)*scale*y

def integrate(bath,*,neutron_order=12,positron_order=20,initial_order=8,outgoing_order=8,phi_order=8,tail=40.,scenarios=None,budget_seconds=110.):
 if not isinstance(bath,PC.crossed.Bath):raise ValueError('Explicit physical npe bath required')
 for z in [neutron_order,positron_order,initial_order,outgoing_order,phi_order]:
  if isinstance(z,bool)or not isinstance(z,int)or z<4:raise ValueError('Integer orders >=4 required')
 if not math.isfinite(tail)or not 20<=tail<=100 or not math.isfinite(budget_seconds)or not 0<budget_seconds<=120:raise ValueError('Finite tail20..100 and budget(0,120]')
 if max((bath.mu_n-MN)/bath.T,(bath.mu_p-MP)/bath.T)>-5 or abs(bath.mu_e)/bath.T>3 or not .2<=bath.T<=4:raise ValueError('Selected dilute baryons eta<=-5,pair-rich |mu_e/T|<=3,T .2..4 only')
 if scenarios is None:scenarios=[dict(name='empty',mode='empty'),dict(name='LTE',mode='FD',mu_nu=bath.mu_n-bath.mu_p-bath.mu_e),dict(name='noneq',mode='FD',mu_nu=bath.mu_n-bath.mu_p-bath.mu_e-.4*bath.T)]
 if not scenarios:raise ValueError('At least one declared radiation scenario required')
 for s in scenarios:
  if s['mode']not in('empty','FD')or(s['mode']=='FD'and(not math.isfinite(s['mu_nu'])or abs(s['mu_nu']/bath.T)>10)):raise ValueError('Empty/common-T FD radiation with bounded |mu/T|<=10 required')
 if len({s['name']for s in scenarios})!=len(scenarios):raise ValueError('Unique scenario names required')
 # Inverse tails include BOTH initial/final baryon and prescribed radiation centers.
 mu_rad=max([s.get('mu_nu',0.)for s in scenarios]+[0.]);knmax=max(0.,bath.mu_n-MN,bath.mu_p-MP)+tail*bath.T+max(0.,mu_rad)
 kemax=max(0.,-bath.mu_e-ME,mu_rad-(MN-MP)-ME)+tail*bath.T
 kn,wn=grid(neutron_order,bath.T,knmax);ke,we=grid(positron_order,bath.T,kemax)
 ci,wi=leggauss(initial_order);co,wo=leggauss(outgoing_order);ph=2*math.pi*np.arange(phi_order)/phi_order
 EE,CI,CO,PH=np.meshgrid(ME+ke,ci,co,ph,indexing='ij');ee=EE.ravel();ci=CI.ravel();co=CO.ravel();ph=PH.ravel();pe=np.sqrt((ee-ME)*(ee+ME));qweight=(we[:,None,None,None]*wi[None,:,None,None]*wo[None,None,:,None]*np.ones((1,1,1,phi_order))*2*math.pi/phi_order).ravel()
 N=len(ee);acc={s['name']:dict(F={k:[]for k in NAMES},B={k:[]for k in NAMES},net=np.zeros(6),entropy=0.,kms=0.)for s in scenarios};start=time.monotonic();density=0.;maxres=0.;maxmass=0.;maxenergy=0.;points=0
 eb=E.Bath(bath.T,MP,MN,bath.mu_p,bath.mu_n,-bath.mu_e,anti=True,ml=ME,Mweak=MW)
 for kinetic,wkin in zip(kn,wn):
  if time.monotonic()-start>budget_seconds:raise TimeoutError('Bounded total positron event integral')
  en=MN+kinetic;pn=math.sqrt(kinetic*(kinetic+2*MN));nvec=np.tile([en,0,0,pn],(N,1));lvec=np.column_stack([ee,pe*np.sqrt(1-ci*ci),np.zeros(N),pe*ci]);P=nvec+lvec;ss=P[:,0]**2-np.sum(P[:,1:]**2,axis=1);rt=np.sqrt(ss);beta=P[:,1:]/P[:,0,None];b2=np.sum(beta*beta,axis=1);axis=beta/np.sqrt(b2)[:,None];e2=np.tile([0,1,0],(N,1));e1=np.cross(e2,axis)
  pi=np.sqrt((ss-(MN+ME)**2)*(ss-(MN-ME)**2))/(2*rt);pf=(ss-MP*MP)/(2*rt);direction=co[:,None]*axis+np.sqrt(1-co*co)[:,None]*(np.cos(ph)[:,None]*e1+np.sin(ph)[:,None]*e2)
  kcm=np.column_stack([pf,pf[:,None]*direction]);pcm=np.column_stack([(ss+MP*MP)/(2*rt),-pf[:,None]*direction]);k=E.boost(kcm,beta);pvec=E.boost(pcm,beta)
  # E.amplitude averages initial p spin in anti+p->n+e+. Reverse n+e+
  # has TWO initial spin averages: divide this value by2, retaining spin2
  # in each incoming density measure (not a new neutrino helicity factor).
  amp=E.amplitude(eb,k,pvec,lvec,nvec)/2
  sigma=amp/(64*math.pi**2*ss)*pf/pi
  flux=pi*rt/(en*ee)
  # spin2 n radial and spin2 e radial+relative-angle, cm azimuth explicit.
  weight=pn*en*wkin/math.pi**2 *pe*ee/(2*math.pi**2)*qweight*flux*sigma/(HBAR*HC**3)
  if np.any(weight<=0)or np.any(~np.isfinite(weight)):raise ArithmeticError('Unresolved physical event weights')
  fn=float(log_expit((bath.mu_n-en)/bath.T));nfn=float(log_expit((en-bath.mu_n)/bath.T));fl=log_expit((-bath.mu_e-ee)/bath.T);nfl=log_expit((ee+bath.mu_e)/bath.T);fp=log_expit((bath.mu_p-pvec[:,0])/bath.T);nfp=log_expit((pvec[:,0]-bath.mu_p)/bath.T)
  density+=pn*en*wkin/math.pi**2/HC**3*math.exp(fn);loss=en+ee-pvec[:,0];energy=np.column_stack([np.ones(N),k[:,0],ee,pvec[:,0],np.full(N,en),loss]);base=np.log(weight);maxres=max(maxres,float(np.max(abs(nvec+lvec-pvec-k))));maxenergy=max(maxenergy,float(np.max(abs(loss-k[:,0]))));maxmass=max(maxmass,float(np.max(abs(pvec[:,0]**2-np.sum(pvec[:,1:]**2,axis=1)-MP*MP)))/MP**2);points+=N
  for s in scenarios:
   a=acc[s['name']]
   if s['mode']=='empty':lf=fn+fl+nfp;lb=None
   else:
    fnu=log_expit((s['mu_nu']-k[:,0])/bath.T);nfnu=log_expit((k[:,0]-s['mu_nu'])/bath.T);lf=fn+fl+nfp+nfnu;lb=nfn+nfl+fp+fnu;delta=lf-lb;expected=(bath.mu_n-bath.mu_p-bath.mu_e-s['mu_nu'])/bath.T;a['kms']=max(a['kms'],float(np.max(abs(delta-expected))))
   for i,name in enumerate(NAMES):
    a['F'][name].append(float(logsumexp(base+lf+np.log(energy[:,i]))))
    if lb is not None:a['B'][name].append(float(logsumexp(base+lb+np.log(energy[:,i]))))
   if lb is None:net=np.exp(base+lf)
   else:
    net=np.sign(delta)*np.exp(base+np.maximum(lf,lb))*(-np.expm1(-abs(delta)));a['entropy']+=float(net@delta)
   a['net']+=net@energy
 out={}
 for s in scenarios:
  a=acc[s['name']];F={k:PC.crossed.represented(float(logsumexp(v)))for k,v in a['F'].items()};B={k:PC.crossed.represented(float(logsumexp(v))if v else None)for k,v in a['B'].items()};net=dict(zip(NAMES,a['net'].tolist()));scale=F['neutrino_energy']['value']+(B['neutrino_energy']['value']or 0.);material=net['proton_energy']-net['neutron_energy']-net['positron_energy']
  out[s['name']]=dict(scenario=s,forward=F,backward=B,net=net,material_energy_change=material,direct_material_energy_change=-net['material_energy_loss'],radiation_energy_change=net['neutrino_energy'],energy_ledger_scaled=(material+net['neutrino_energy'])/scale,direct_ledger_scaled=(net['neutrino_energy']-net['material_energy_loss'])/scale,number_sources=dict(n=-net['number'],p=net['number'],positron=-net['number'],electron_net=net['number'],antinue=net['number']),matter_Le_change=net['number'],radiation_Le_change=-net['number'],entropy=a['entropy']if s['mode']=='FD'else None,max_event_kms_error=a['kms'])
 return dict(bath=vars(bath),scenarios=out,neutron_density=density,controls=dict(neutron_order=neutron_order,positron_order=positron_order,initial_order=initial_order,outgoing_order=outgoing_order,phi_order=phi_order,tail=tail,knmax=knmax,positron_kinetic_max=kemax),max_four_residual=maxres,max_mass_residual=maxmass,max_single_event_energy_residual=maxenergy,points=points,seconds=time.monotonic()-start,units='number fm^-3 s^-1; energy MeV fm^-3 s^-1; entropy kB fm^-3 s^-1',inventory='n+e+ <-> p+antinue ONLY; no beta decay or muons/meanfields')
