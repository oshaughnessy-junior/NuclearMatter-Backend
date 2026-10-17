"""Original delta-shell references, no candidate or source EOS imports."""
from pathlib import Path
import importlib.util,sys,math,time,json
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import log_expit,logsumexp
P=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('spectral_review_unique_pure_primitives',P/'primitives.py');V=importlib.util.module_from_spec(s);sys.modules[s.name]=V;s.loader.exec_module(V)
MN,MP,ME,MW,HC,HBAR,b=V.MN,V.MP,V.ME,V.MW,V.HC,V.HBAR,V.bath

def differential(channel,E,El,c,order=20,phi=6,tail=65.):
 pe=math.sqrt((El-ME)*(El+ME));w=E-El;q=math.sqrt((E-pe)**2+2*E*pe*(1-c));invariant=w*w-q*q
 if q<=0 or abs(invariant)/(q*q+w*w)<V.LOCK['gates']['lightcone_conditioning_min']:raise ArithmeticError('Unresolved q0/lightcone coordinate; not physicalzero')
 ma,mb,mua,mub,mul=(MN,MP,b.mu_n,b.mu_p,b.mu_e)if channel=='EC'else(MP,MN,b.mu_p,b.mu_n,-b.mu_e)
 x,g=leggauss(order);points=[];weights=[];kind='spacelike_originalpz'
 if invariant<0:
  rt=math.sqrt(-invariant);z0=-(mb*mb-ma*ma-invariant)/(2*rt);emin=(q*math.hypot(ma,z0)+w*z0)/rt;upper=max(emin,mua,mub-w)+tail*b.T
  def u_at(ea):return(((ea*rt-w*z0)/q)**2-ma*ma-z0*z0)/(2*ma)
  top=u_at(upper)
  if top<=0:raise ArithmeticError('Unresolved positive transverse phase interval')
  cuts=sorted({0.,top}|{u_at(center+h*b.T)for center in[emin,mua,mub-w]for h in[0,2,6,12,24,40]if emin<center+h*b.T<upper})
  for a,d in zip(cuts[:-1],cuts[1:]):
   if not d>a:raise ArithmeticError('Unresolved transformedpanel; no widthonlyskip')
   points.extend(a+(x+1)*(d-a)/2);weights.extend(g*(d-a)/2)
  u=np.repeat(points,phi);wt=np.repeat(weights,phi)/phi;ph=np.tile(2*math.pi*np.arange(phi)/phi,len(points));r=np.sqrt(2*ma*u);ep=np.sqrt(ma*ma+r*r+z0*z0);pz=(q*z0+w*ep)/rt;ea=(q*ep+w*z0)/rt;eb=ea+w;jac=abs((pz+q)/eb-pz/ea)
  if np.any(jac<=0):raise ArithmeticError('Unresolved delta Jacobian')
  phase=wt*ma/(ea*eb*jac)
 else:
  kind='timelike_Qrest_shell';star=math.copysign(math.sqrt(invariant),w);east=(mb*mb-ma*ma-invariant)/(2*star);ebst=east+star
  # Exact kinematic no-shell witness, NOT finite-tail empty search.
  if east<ma or ebst<mb:return dict(logj=None,logchi=None,exact_empty=True,support=dict(kind=kind,Ea_star=east,Eb_star=ebst,masses=[ma,mb]),points=0)
  rad2=(east-ma)*(east+ma)
  if rad2<=0:raise ArithmeticError('Unresolved collapsed shell boundary')
  r0=math.sqrt(rad2);v=q/w;gamma=abs(w/star);slope=gamma*v*r0;center=gamma*east;emin=center-abs(slope);upper=center+abs(slope)
  cuts=sorted({-1.,1.}|{c for energy in[emin,mua,mub-w]for h in[0,2,6,12,24,40,tail]if -1<(c:=(energy+h*b.T-center)/slope)<1})
  cs=[];cw=[]
  for left,right in zip(cuts[:-1],cuts[1:]):
   if not right>left:raise ArithmeticError('Unresolvedtimelikeangularpanel; no widthonlyskip')
   cs.extend(left+(x+1)*(right-left)/2);cw.extend(g*(right-left)/2)
  cc=np.repeat(cs,phi);wt=np.repeat(cw,phi)/phi;ph=np.tile(2*math.pi*np.arange(phi)/phi,len(cs));ea=gamma*(east+v*r0*cc);pz=gamma*(r0*cc+v*east);r=r0*np.sqrt(1-cc*cc);eb=ea+w;phase=wt*r0/abs(star)
 pa=np.column_stack([ea,r*np.cos(ph),r*np.sin(ph),pz]);pb=pa+np.array([w,0.,0.,q]);kz=(E*E+q*q-pe*pe)/(2*q);kx2=E*E-kz*kz
 if kx2 < -1e-10*E*E:raise ArithmeticError('Lepton q-axis shell unresolved')
 kx=math.sqrt(max(0.,kx2));k=np.tile([E,kx,0.,kz],(len(ea),1));l=np.tile([El,kx,0.,kz-q],(len(ea),1));amp=V.EC.scalar_amplitude(k,pa,l,pb)if channel=='EC'else V.PC.amplitude(k,pa,l,pb)
 if np.any(amp<=0)or np.any(phase<=0)or not np.all(np.isfinite(amp))or not np.all(np.isfinite(phase)):raise ArithmeticError('Positive current/phase unresolved')
 lf=log_expit((mub-eb)/b.T)+log_expit((ea-mua)/b.T)+log_expit((mul-El)/b.T);lb=log_expit((mua-ea)/b.T)+log_expit((eb-mub)/b.T)+log_expit((El-mul)/b.T);base=np.log(phase*amp*pe/(32*math.pi**3*E*HBAR))
 shell=max(float(np.max(abs(V.EC.dot(pa,pa)-ma*ma)))/ma**2,float(np.max(abs(V.EC.dot(pb,pb)-mb*mb)))/mb**2)
 return dict(logj=float(logsumexp(base+lf)),logchi=float(logsumexp(base+lb)),exact_empty=False,support=dict(kind=kind,invariant=invariant,Ea_min=emin,Ea_upper=upper),shell=shell,points=len(ea),controls=dict(order=order,phi=phi,tail=tail))

def integrated(channel,E,order=6,qorder=10,inner=20,phi=6,tail=65.,outer_tail=40.,budget=105.):
 start=time.monotonic();mid=E+(MN-MP if channel=='EC'else MP-MN);mu=b.mu_e if channel=='EC'else-b.mu_e;upper=max(mid,ME,mu)+outer_tail*b.T
 cuts=sorted({ME,upper,math.hypot(E,ME)}|{v for center in[mid,mu]for h in[-12,-6,-3,-1.5,-.75,-.25,0,.25,.75,1.5,3,6,12]if ME<(v:=center+h*b.T)<upper});x,wg=leggauss(order);z,zw=leggauss(qorder);logs={'j':[],'chi':[]};calls=timelike=empty=0
 for lo,hi in zip(cuts[:-1],cuts[1:]):
  lp=lambda e:math.log1p(math.sqrt((e-ME)*(e+ME))/b.T);a,d=lp(lo),lp(hi)
  if not d>a:raise ArithmeticError('Collapsedoutermappedpanel; unresolved')
  for y,ww in zip(a+(x+1)*(d-a)/2,wg*(d-a)/2):
   pe=b.T*math.expm1(y);el=math.hypot(pe,ME);omega=E-el;qlo=abs(E-pe);qhi=E+pe;qedges=sorted({qlo,qhi}|{r for r in[abs(omega),math.sqrt(max(0.,omega*omega-(MN-MP)**2))]if qlo<r<qhi})
   for qa,qb in zip(qedges[:-1],qedges[1:]):
    for q,wq in zip(qa+(z+1)*(qb-qa)/2,zw*(qb-qa)/2):
     if time.monotonic()-start>budget:raise TimeoutError('Independentphase105s; incomplete no coefficientreturned')
     c=(E*E+pe*pe-q*q)/(2*E*pe)
     if abs(c)>1:raise ArithmeticError('Outgoing angle shell unresolved')
     r=differential(channel,E,el,float(c),inner,phi,tail);calls+=1;timelike+=r['support']['kind']=='timelike_Qrest_shell';empty+=r['exact_empty'];weight=ww*b.T*math.exp(y)*wq*q/(E*el)
     if weight<=0:raise ArithmeticError('Positiveouterweightunresolved')
     if not r['exact_empty']:
      for key in logs:logs[key].append(r['log'+key]+math.log(weight))
  (P/f'partial_capture_{channel}_{E}_{order}.json').write_text(json.dumps(dict(complete=False,channel=channel,E=E,completed_outer_edge=hi,calls=calls,seconds=time.monotonic()-start,scope='partial evidence only; no coefficientadmission'),indent=2)+'\n')
 if not logs['j']or not logs['chi']:raise ArithmeticError('Positive integrated channel unresolved; notexactempty')
 return dict(E=E,channel=channel,logj=float(logsumexp(logs['j'])),logchi=float(logsumexp(logs['chi'])),controls=dict(order=order,qorder=qorder,inner=inner,phi=phi,tail=tail,outer_tail=outer_tail),calls=calls,timelike=timelike,exact_empty_differentials=empty,seconds=time.monotonic()-start,upperleptonenergy=upper)
