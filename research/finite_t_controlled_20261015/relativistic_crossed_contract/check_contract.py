from pathlib import Path
import sys,json,math
from scipy.special import log_expit
sys.dont_write_bytecode=True
from beta import *
checks=[];rows=[]
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
def dot(a,b):return np.sum(a*b*MET,axis=1)
e=np.array([.63,.81,1.12]);c=np.array([-.61,.19,.73]);pn,pp,l,k,*_=electron_events(e,c,MN,MP,ME);v=1.;a=1.2723;X=dot(k,pn)*dot(l,pp);Y=dot(k,pp)*dot(l,pn);point=16*GF**2*VUD**2*((v*v+a*a)*(X+Y)+2*v*a*(X-Y)-(v*v-a*a)*MN*MP*dot(k,l));spin=amplitude(pn,pp,l,k,model='point_VA');gate('direct crossed spinors vs scalar pointVA invariant',np.max(abs(spin/point-1)),3e-13)
# Free support is a Lorentz-invariant bound, not a temperature threshold.
for ml,label in [(ME,'electron'),(MMU,'muon')]:
 for initial,final,name in [(MN,MP,'neutron decay'),(MP,MN,'proton decay')]:
  allowed=initial>final+ml;rows.append(dict(branch=name,lepton=label,allowed_free_vacuum=allowed,Qmax_squared=(initial-final)**2,lepton_mass_squared=ml*ml))
  gate(label+' '+name+' support',float(allowed!=(name=='neutron decay'and label=='electron')),0.)
for T,chem in [(1.7,(930,933,2)),(.23,(939,941,.6))]:
 mup,mun,mue=chem;Q=k+l;fp=log_expit((mup-pp[:,0])/T);nfp=log_expit((pp[:,0]-mup)/T);fn=log_expit((mun-pn[:,0])/T);nfn=log_expit((pn[:,0]-mun)/T);fe=log_expit((mue-l[:,0])/T);nfe=log_expit((l[:,0]-mue)/T)
 logWpn=fp+nfn;logWnp=fn+nfp;gate(f'{T} crossed baryon KMS',np.max(abs(logWnp-logWpn+(Q[:,0]+mup-mun)/T)),7e-13)
 gate(f'{T} electron incoming vs outgoing products',np.max(abs(nfe-fe-(l[:,0]-mue)/T)),7e-13)
# Explicitly rejected forbidden branches and unsupported/invalid controls.
for label,fn in [('muon vacuum',lambda:pair_integral(8,ml=MMU)),('proton beta+',lambda:electron_integral(8,mn=MP,mp=MN)),('noninteger order',lambda:pair_integral(4.5)),('bool order',lambda:pair_integral(True)),('nonfinite mass',lambda:pair_integral(mn=float('nan'))),('negative mass',lambda:pair_integral(mp=-1)),('bad current',lambda:pair_integral(model='capture')),('endpoint electron spectrum',lambda:spectrum(ME)),('nonfinite spectrum',lambda:spectrum(float('nan')))]:
 try:fn();rejected=False
 except (ValueError,ArithmeticError):rejected=True
 gate(label,float(not rejected),0.)
for label,kw in [('negative weak mass',dict(Mweak=-1)),('nonfinite weak mass',dict(Mweak=float('nan'))),('invalid inverse flag',dict(inverse=1))]:
 try:amplitude(pn,pp,l,k,**kw);rejected=False
 except (ValueError,ArithmeticError):rejected=True
 gate(label,float(not rejected),0.)
Path(__file__).with_name('contract_results.json').write_text(json.dumps(dict(passed=all(c['passed']for c in checks),checks=checks,rows=rows),indent=2)+'\n');print(len(checks),all(c['passed']for c in checks))
