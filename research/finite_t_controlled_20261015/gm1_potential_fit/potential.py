"""Restricted analytic GM1 Helmholtz per-baryon potential, not an EOS replacement."""
from __future__ import annotations
from pathlib import Path
import json,math
import numpy as np
from numpy.polynomial import chebyshev as ch
P=Path(__file__).resolve().parent
DOMAIN=dict(n=(.18,.26),Y=(.18,.33),T=(0.,3.))
M=939.

def scale(x,bounds):return (2*x-sum(bounds))/(bounds[1]-bounds[0])

def isospin_coordinate(Y):
 bounds=((1-2*DOMAIN['Y'][1])**2,(1-2*DOMAIN['Y'][0])**2)
 return scale((1-2*Y)**2,bounds)

def field(c,n,Y,dn=0,dy=0):
 a=np.asarray(c,float)
 if dn:a=ch.chebder(a,m=dn,axis=0)*(2/(DOMAIN['n'][1]-DOMAIN['n'][0]))**dn
 x=scale(n,DOMAIN['n']);z=isospin_coordinate(Y)
 if dy==0:return float(ch.chebval2d(x,z,a))
 span=(1-2*DOMAIN['Y'][0])**2-(1-2*DOMAIN['Y'][1])**2
 zY=-8*(1-2*Y)/span;zYY=16/span
 first=float(ch.chebval2d(x,z,ch.chebder(a,axis=1)))
 if dy==1:return first*zY
 if dy==2:return float(ch.chebval2d(x,z,ch.chebder(a,m=2,axis=1)))*zY*zY+first*zYY
 raise ValueError('Only first/second Y derivatives supplied')


def evaluate(n,T,Y,coefficients=None):
 if not all(math.isfinite(x) for x in (n,T,Y)) or not(DOMAIN['n'][0]<=n<=DOMAIN['n'][1] and DOMAIN['Y'][0]<=Y<=DOMAIN['Y'][1] and DOMAIN['T'][0]<=T<=DOMAIN['T'][1]):raise ValueError('Outside explicit finite potential-fit domain')
 c=json.loads((P/'coefficients.json').read_text())['fields'] if coefficients is None else coefficients
 def deriv(dn=0,dy=0,dt=0):
  v=0.
  for key,power in [('cold',0),('T2',2),('T4',4)]:
   if power<dt:continue
   factor=math.factorial(power)/math.factorial(power-dt)*T**(power-dt)
   v+=factor*field(c[key],n,Y,dn,dy)
  if dn==dy==dt==0:v+=M
  return v
 a=deriv();an=deriv(dn=1);ay=deriv(dy=1);at=deriv(dt=1);ann=deriv(dn=2);any_=deriv(dn=1,dy=1);ayy=deriv(dy=2)
 H=np.array([[2*an+n*ann-2*Y*any_+Y*Y/n*ayy,2*an+n*ann+(1-2*Y)*any_-Y*(1-Y)/n*ayy],
             [2*an+n*ann+(1-2*Y)*any_-Y*(1-Y)/n*ayy,2*an+n*ann+2*(1-Y)*any_+(1-Y)**2/n*ayy]])
 return dict(free_per_baryon=a,free_energy_density=n*a,entropy_density=-n*at,energy_density=n*(a-T*at),pressure=n*n*an,
             mu_n=a+n*an-Y*ay,mu_p=a+n*an+(1-Y)*ay,cv_fixed_nY=-n*T*deriv(dt=2),chemical_hessian=H.tolist(),min_chemical_eigenvalue=float(np.linalg.eigvalsh(H)[0]))
