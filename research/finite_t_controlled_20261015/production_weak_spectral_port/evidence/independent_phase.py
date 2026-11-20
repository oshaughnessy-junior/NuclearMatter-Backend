"""Original longitudinal-delta integration; alternate transverse-momentum map.

Independent forward phase coordinate, same source-model literal Dirac current.
"""
from pathlib import Path
import sys,json,math
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'test_mirror/src'))
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import logsumexp
from nmbackend.reference import _weak_current as C
from nmbackend.reference import _weak_quadrature as Q
from nmbackend.reference import weak_reference as W
lock=json.loads((R/'LOCK.json').read_text());bath=W.FreeElectronBath(**lock['bath'])
results=[]
for ch,E,el,cos in [lock['new_differential'][0],lock['new_differential'][2]]:
 ma,mb=(C.MN,C.MP)if ch=='EC'else(C.MP,C.MN);mua,mub=(bath.mu_n,bath.mu_p)if ch=='EC'else(bath.mu_p,bath.mu_n)
 pe=math.sqrt((el-C.ME)*(el+C.ME));w=E-el;q=math.sqrt((E-pe)**2+2*E*pe*(1-cos));aa=q*q-w*w;cc=(ma-mb)*(ma+mb)-aa
 assert aa>0
 kZ=(E*E+q*q-pe*pe)/(2*q);kX=math.sqrt(max(0.,E*E-kZ*kZ))
 finest=None;profiles=[]
 for order,tail,phiN in[(36,50,12),(56,70,16)]:
  x,wt=leggauss(order);pm=math.sqrt(tail*bath.T*(2*ma+tail*bath.T));pt=(x+1)*pm/2;wp=wt*pm/2
  z=(q*cc+w*np.sqrt(cc*cc+4*aa*(ma*ma+pt*pt)))/(2*aa);ea=np.sqrt(ma*ma+pt*pt+z*z);eb=ea+w
  derivative=(z+q)/eb-z/ea
  phi=2*math.pi*np.arange(phiN)/phiN
  a=np.column_stack([np.repeat(ea,phiN),(pt[:,None]*np.cos(phi)).ravel(),(pt[:,None]*np.sin(phi)).ravel(),np.repeat(z,phiN)])
  b=a+np.array([w,0,0,q]);k=np.tile([E,kX,0,kZ],(len(a),1));lepton=np.tile([el,kX,0,kZ-q],(len(a),1))
  n,p=(a,b)if ch=='EC'else(b,a)
  amp=C.literal_capture(ch,k,n,lepton,p)
  weight=wp*pt/(ea*eb*abs(derivative))/phiN
  base=np.log(np.repeat(weight,phiN))+np.log(amp)+math.log(pe/(32*math.pi**3*E*C.HBAR))
  eaa=a[:,0];mu=bath.mu_e if ch=='EC'else-bath.mu_e
  lf=Q.log_fd((mub-eaa-w)/bath.T)+Q.log_fd((eaa-mua)/bath.T)+Q.log_fd((mu-el)/bath.T)
  lb=Q.log_fd((mua-eaa)/bath.T)+Q.log_fd((eaa+w-mub)/bath.T)+Q.log_fd((el-mu)/bath.T)
  vals=[float(logsumexp(base+lf)),float(logsumexp(base+lb))];profiles.append(dict(order=order,tail=tail,phi=phiN,logs=vals))
 ref=W.capture_differential_coefficients(ch,bath,E,el,cos,W.SpectralControls.fine())
 errors=[abs(math.expm1(profiles[-1]['logs'][i]-v))for i,v in enumerate([ref.j.log,ref.true_chi.log])]
 refine=[abs(math.expm1(profiles[0]['logs'][i]-profiles[1]['logs'][i]))for i in(0,1)]
 results.append(dict(point=[ch,E,el,cos],profiles=profiles,candidate_logs=[ref.j.log,ref.true_chi.log],errors=errors,refinement=refine,gate=3e-10,passed=max(errors+refine)<3e-10))
(R/'evidence/independent_phase.json').write_text(json.dumps(results,indent=2)+'\n');print([(r['point'],r['errors'],r['refinement'],r['passed'])for r in results])
