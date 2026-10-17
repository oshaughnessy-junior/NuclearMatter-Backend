"""Independent threshold roots, vacuum reciprocity and point-VA absolute limit."""
from pathlib import Path
import json,math
import numpy as np
from scipy.optimize import brentq
from numpy.polynomial.legendre import leggauss
import electron_capture as C
HERE=Path(__file__).resolve().parent;checks=[]
def ck(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(np.isfinite(error)and abs(error)<=tol)))
bath,state=C.matched();D=(C.MN*C.MN-C.MP*C.MP-C.ME*C.ME)/2
thresholds=[]
for P in [0.,57.,184.,400.]:
 Ep=math.hypot(C.MP,P)
 for c in [-.83,.17,.72]:
  root=brentq(lambda ee:Ep*ee-P*math.sqrt(ee*ee-C.ME*C.ME)*c-D,C.ME,20.,xtol=1e-13)
  analytic=float(C.threshold(Ep,P,c));ck('independent threshold root '+str((P,c)),analytic/root-1,3e-12)
  thresholds.append(dict(P=P,cosine=c,root=root,analytic=analytic))
rest_threshold=D/C.MP;ck('rest threshold totalEe',float(C.threshold(C.MP,0.,.3))/rest_threshold-1,3e-15)
ck('movingtarget threshold not a fixed globalEe cut',float(not float(C.threshold(math.hypot(C.MP,400.),400.,-.83))<rest_threshold),0.)

def vacuum_events(ee,c,mn=C.MN,mp=C.MP):
 c=np.asarray(c);pe=math.sqrt(ee*ee-C.ME*C.ME);p=np.tile([mp,0,0,0.],(len(c),1));l=np.tile([ee,0,0,pe],(len(c),1));tot=p+l;s=C.dot(tot,tot);root=np.sqrt(s);beta=tot[:,1:]/tot[:,0,None];pf=(s-mn*mn)/(2*root);pi=np.sqrt((s-(mp+C.ME)**2)*(s-(mp-C.ME)**2))/(2*root)
 k=C.E.boost(np.column_stack([pf,pf*np.sqrt(1-c*c),np.zeros(len(c)),pf*c]),beta);n=p+l-k
 return k,n,l,p,s,pi,pf
vacuum=[];c,w=leggauss(32)
for ee in [1.31,3.7,11.9]:
 k,n,l,p,s,pi,pf=vacuum_events(ee,c);ec=C.spinor_amplitude(k,n,l,p)
 b=C.E.Bath(bath.T,C.MN,C.MP,bath.mu_n,bath.mu_p,bath.mu_e,ml=C.ME,Mweak=C.MW);absorb=C.E.amplitude(b,k,n,l,p)
 sigma_ec=float(np.sum(w*ec/(64*math.pi**2*s)*pf/pi)*2*math.pi)
 sigma_abs=float(np.sum(w*absorb/(64*math.pi**2*s)*pi/pf)*2*math.pi)
 expected=2*(pi[0]/pf[0])**2*sigma_ec
 ck('vacuum spin/phase reciprocity '+str(ee),sigma_abs/expected-1,3e-12)
 vacuum.append(dict(Ee=ee,EC_sigma_MeV_minus2=sigma_ec,inverse_capture_sigma_MeV_minus2=sigma_abs,ratio=sigma_abs/sigma_ec))

def pointVA(k,n,l,p,mn,mp):
 # Explicit trace/spinors in a declared point-VA LIMIT model, no formfactor fit.
 un=C.E.spinors(n,mn);up=C.E.spinors(p,mp);uk=C.E.spinors(k,0);ul=C.E.spinors(l,C.ME);V=C.E.G-1.2723*(C.E.G@C.E.G5);V=np.broadcast_to(V,(len(k),4,4,4));V=C.E.adj(V)
 left=C.E.G@(C.E.I-C.E.G5);J=np.einsum('nai,ab,nubc,ncj->nuij',un.conj(),C.E.G[0],V,up).reshape((-1,4,4));L=np.einsum('nai,ab,ubc,ncj->nuij',uk.conj(),C.E.G[0],left,ul).reshape((-1,4,4));a=np.einsum('nui,u,nuj->nij',L,C.E.MET,J)
 return C.E.GF**2*C.E.VUD**2/8*np.sum(abs(a)**2,axis=(1,2))
heavy=[];delta=C.MN-C.MP
for ee in [5.,10.]:
 previous=None
 for mass in [1e4,1e5,1e6]:
  mn=mass+delta/2;mp=mass-delta/2;k,n,l,p,s,pi,pf=vacuum_events(ee,c,mn,mp);amp=pointVA(k,n,l,p,mn,mp)
  actual=float(np.sum(w*amp/(64*math.pi**2*s)*pf/pi)*2*math.pi*math.sqrt(ee*ee-C.ME*C.ME)/ee)
  ref=C.E.GF**2*C.E.VUD**2/(2*math.pi)*(1+3*1.2723**2)*(ee-delta)**2;error=actual/ref-1
  if previous is not None:ck('pointVA heavyM convergence '+str((ee,mass)),abs(error/previous),.2)
  if mass==1e6:ck('absolute allowed EC sigma*v '+str(ee),error,1e-4)
  previous=error;heavy.append(dict(Ee=ee,M=mass,actual_sigma_v_MeV_minus2=actual,allowed=ref,relative_error=error,scope='Mathematical pointVA heavyM normalization limit; not actualfullcurrent thermal source'))

bad=[('nonfinitebath',lambda:C.Bath(2.,np.nan,919.,0.)),('empty_scenario_mu',lambda:C.integrate(bath,scenarios=[dict(name='bad',mode='empty',mu_nu=0.)])),('different_radT',lambda:C.integrate(bath,scenarios=[dict(name='bad',mode='FD',mu_nu=0.,temperature=2.1)])),('nonfinite_mu_nu',lambda:C.integrate(bath,scenarios=[dict(name='bad',mode='FD',mu_nu=np.nan)])),('ignored_extra_key',lambda:C.integrate(bath,scenarios=[dict(name='bad',mode='empty',nonsense=1)])),('bool_order',lambda:C.integrate(bath,proton_order=True)),('negative_tail',lambda:C.integrate(bath,tail=-1.)),('negative_budget',lambda:C.integrate(bath,budget_seconds=-1.)),('degenerate_baryon',lambda:C.integrate(C.Bath(2.,C.MN,C.MP,0.)))]
for name,fn in bad:
 try:fn();ok=False
 except ValueError:ok=True
 ck('reject '+name,float(not ok),0)
out=dict(checks=checks,passed=sum(x['passed']for x in checks),failed=[x for x in checks if not x['passed']],thresholds=thresholds,rest_electron_total_threshold=rest_threshold,vacuum=vacuum,heavy=heavy)
(HERE/'controls_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k]for k in ['passed','failed']}))
