"""New lab-Enu conditional event spectrum versus independent tensor spectrum.

Delta in isotropic rest neutrino orientation: dc*/2 -> 1/(2 gamma v E*).
"""
from pathlib import Path
import json,math,time
import numpy as np
from scipy.special import roots_genlaguerre,log_expit,logsumexp
from numpy.polynomial.legendre import leggauss
import moments as M
P=Path(__file__).resolve().parent
def conditional(bath,E_lab=.42,order=10,parent_order=12,phi_order=8):
 x,w=leggauss(order);xr,wr=roots_genlaguerre(parent_order,.5);phi=2*math.pi*np.arange(phi_order)/phi_order;logs=dict(j=[],chi=[]);maxshell=0.;start=time.monotonic()
 endpoint=(M.K.MN-M.K.MP-M.K.ME)*(M.K.MN+M.K.MP+M.K.ME)/(2*M.K.MN)
 for xx,ww in zip(xr,wr):
  En=M.K.MN+bath.T*xx;p=math.sqrt(bath.T*xx*(2*M.K.MN+bath.T*xx));gamma=En/M.K.MN;v=p/En
  parent=ww*bath.T*math.sqrt(bath.T*(2*M.K.MN+bath.T*xx))*En/(math.pi**2*M.HC**3)
  lo=max(0.,E_lab/(gamma*(1+v)));hi=min(endpoint,E_lab/(gamma*(1-v)))
  if hi<=lo:continue
  for ex,ew in zip(x,w):
   enu=lo+(ex+1)*(hi-lo)/2;enuweight=ew*(hi-lo)/2;cnu=(E_lab/(gamma*enu)-1)/v;assert abs(cnu)<1
   S=M.K.MN*M.K.MN-2*M.K.MN*enu;rt=math.sqrt(S);Es=(S+M.K.ME*M.K.ME-M.K.MP*M.K.MP)/(2*rt);ps=math.sqrt((Es-M.K.ME)*(Es+M.K.ME));gm=(M.K.MN-enu)/rt;bv=enu/(M.K.MN-enu);a,b=gm*(Es-bv*ps),gm*(Es+bv*ps)
   Ee=a+(x+1)*(b-a)/2;we=w*(b-a)/2;pe=np.sqrt((Ee-M.K.ME)*(Ee+M.K.ME));c=(M.K.MN*M.K.MN+M.K.ME*M.K.ME-M.K.MP*M.K.MP-2*M.K.MN*(Ee+enu)+2*Ee*enu)/(2*pe*enu)
   assert np.max(abs(c))<1+1e-7
   pn=np.tile([M.K.MN,0.,0.,0.],(order,1));nu=np.tile([enu,0.,0.,enu],(order,1));e=np.column_stack([Ee,pe*np.sqrt(np.maximum(0.,1-c*c)),np.zeros(order),pe*c]);pp=pn-e-nu
   amp=M.B.amplitude(pn,pp,e,nu);base=we*amp/(64*math.pi**3*M.K.MN*M.B.HBAR)*enuweight/(2*gamma*v*enu)*parent/gamma
   assert np.all(base>0)
   si=math.sqrt(1-cnu*cnu)
   for ph in phi:
    # Rotate the rest neutrino +z into the selected lab-energy cone.
    def energy(v4):return gamma*(v4[:,0]+v*(-si*(v4[:,1]*math.cos(ph)-v4[:,2]*math.sin(ph))+cnu*v4[:,3]))
    Ep=energy(pp);El=energy(e);Ev=energy(nu);maxshell=max(maxshell,float(np.max(abs(En-Ep-El-Ev))),float(np.max(abs(Ev-E_lab))))
    fn=float(log_expit((bath.mu_n-En)/bath.T));nfn=float(log_expit((En-bath.mu_n)/bath.T));fp=log_expit((bath.mu_p-Ep)/bath.T);nfp=log_expit((Ep-bath.mu_p)/bath.T);fe=log_expit((bath.mu_e-El)/bath.T);nfe=log_expit((El-bath.mu_e)/bath.T)
    logs['j'].append(float(logsumexp(np.log(base)+xx-math.log(phi_order)+fn+nfp+nfe)))
    logs['chi'].append(float(logsumexp(np.log(base)+xx-math.log(phi_order)+nfn+fp+fe)))
 dos=E_lab*E_lab/(2*math.pi**2*M.HC**3)
 return dict(Enu=E_lab,j=M.represented(float(logsumexp(logs['j']))-math.log(dos)),chi=M.represented(float(logsumexp(logs['chi']))-math.log(dos)),DOS=dos,max_conditional_energy_residual=maxshell,seconds=time.monotonic()-start,controls=dict(order=order,parent_order=parent_order,phi_order=phi_order),derivation='Full rest Dalitz dGamma/(dEanti*dEe) averaged parent spin; conditional isotropic cone Jacobian1/(2gamma v Eanti*); parent spin2 and proper time; divide one-helicity DOS only for spectral coefficient comparison')

if __name__=='__main__':
 bath,material=M.K.matched_npe(.03,1e-12,.3);runs=[conditional(bath,order=o,parent_order=r,phi_order=p)for o,r,p in[(6,8,4),(10,12,8),(16,16,12)]]
 tensor=M.K.spectral(bath,.42,angle_order=14,lepton_order=16,widths=12.,energy_order=24,phi_order=4,tail=70.)
 checks=[]
 for name in('j','chi'):
  for i in(0,1):
   e=math.expm1(runs[i][name]['log']-runs[-1][name]['log']);checks.append(dict(name='conditional_refinement_'+name+'_'+str(i),error=abs(e),limit=3e-6,passed=abs(e)<=3e-6))
  e=math.expm1(runs[-1][name]['log']-tensor[name]['log']);checks.append(dict(name='new_tensor_spectrum_'+name,error=abs(e),limit=3e-6,passed=abs(e)<=3e-6))
 out=dict(material=material,runs=runs,tensor=tensor,checks=checks,passed=sum(c['passed']for c in checks),failed=sum(not c['passed']for c in checks),scope='New selected Enu .42 spectral anchor only; not full-Enu tensor integration')
 (P/'spectrum_results.json').write_text(json.dumps(out,indent=2));print(out['passed'],out['failed'])
 if out['failed']:raise SystemExit(1)
