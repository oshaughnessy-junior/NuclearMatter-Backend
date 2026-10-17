"""Independent value-only Chebyshev recurrence checks; algebra, not EOS accuracy."""
import json,math
import numpy as np
from potential import *
c=json.loads((P/'coefficients.json').read_text())['fields'];checks=[]
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(abs(error)<=tol)))
def rec(x):
 a=[1,x]
 for k in range(2,7):a.append(2*x*a[-1]-a[-2])
 return a
def a(n,T,Y):
 xx=rec((n-.22)/.04);z=rec((2*(1-2*Y)**2-(.34**2+.64**2))/(.64**2-.34**2));v=939.
 for key,pow in [('cold',0),('T2',2),('T4',4),('T6',6)]:v+=T**pow*sum(c[key][i][j]*xx[i]*z[j]for i in range(7)for j in range(7))
 return v
def free(nn,pp,T):return (nn+pp)*a(nn+pp,T,pp/(nn+pp))
def mu(pair,T):
 vals=[]
 for k in range(2):
  z=pair.astype(complex);z[k]+=1e-20j;vals.append(np.imag(free(*z,T))/1e-20)
 return np.array(vals)
for n,T,Y in [(.2137,.151,.2179),(.2583,2.781,.3281),(.1857,1.173,.3017)]:
 p=evaluate(n,T,Y);pair=np.array([n*(1-Y),n*Y]);m=mu(pair,T)
 gate('manual value',a(n,T,Y)-p['free_per_baryon'],2e-10);gate('complex mu',max(abs(m-[p['mu_n'],p['mu_p']])),2e-9)
 gate('complex pressure',n*n*np.imag(a(n+1e-20j,T,Y))/1e-20-p['pressure'],2e-10)
 gate('complex entropy',-n*np.imag(a(n,T+1e-20j,Y))/1e-20-p['entropy_density'],2e-11)
 for frac in [2e-4,1e-4]:
  H=np.zeros((2,2))
  for k in range(2):
   h=pair[k]*frac;pp=pair.copy();mm=pair.copy();pp[k]+=h;mm[k]-=h;H[:,k]=(mu(pp,T)-mu(mm,T))/(2*h)
  gate('complex/FD Hessian',np.linalg.norm(H-p['chemical_hessian'])/np.linalg.norm(H),2e-7)
 gate('Euler',p['energy_density']+p['pressure']-T*p['entropy_density']-sum(pair*m),2e-10)
 h=T*1e-4;sp=-n*np.imag(a(n,T+h+1e-20j,Y))/1e-20;sm=-n*np.imag(a(n,T-h+1e-20j,Y))/1e-20;cv=T*(sp-sm)/(2*h)
 gate('entropy derivative Cv',abs(cv/p['cv_fixed_nY']-1),2e-7)
out=dict(passed=all(z['passed']for z in checks),checks=checks,meaning='structural potential identities, not independent physical EOS validation');(P/'structure_results.json').write_text(json.dumps(out,indent=2)+'\n');print(len(checks),out['passed'])
