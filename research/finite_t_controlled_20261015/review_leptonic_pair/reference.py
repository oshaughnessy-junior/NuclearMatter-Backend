"""Independent angle-polynomial/separatedFD moments; no author thermal routine."""
import math
import numpy as np
from scipy.integrate import quad
from scipy.special import expit
GF=1.1663785e-11;HC=197.3269804;HBAR=6.582119569e-22;ME=.51099895069

def separated(T,mu,v,a,m=ME,tail=90.):
 def moments(chemical):
  upper=math.sqrt(max(0.,chemical-m)+tail*T)
  return [quad(lambda u:2*u*u*math.sqrt(u*u+2*m)*(m+u*u)**j*expit((chemical-m-u*u)/T),0,upper,epsabs=1e-16,epsrel=3e-12,limit=150)[0]for j in range(4)]
 minus,plus=moments(mu),moments(-mu);A={(0,0):16*m**4/3,(1,1):8*m*m,(2,2):16/3,(2,0):-4*m*m/3,(0,2):-4*m*m/3};C=v*v+a*a;D=2*v*v-4*a*a;B={k:C*z for k,z in A.items()};B[0,0]+=2*m**4*D;B[1,1]+=2*m*m*D
 def product(poly,di=0,dj=0):return sum(z*minus[i+di]*plus[j+dj]for(i,j),z in poly.items())
 pref=GF*GF/(24*math.pi**5*HBAR*HC**3);number=pref*product(B);e=pref*product(B,1);p=pref*product(B,0,1);difference=pref*v*a*(product(A,1)-product(A,0,1))
 return dict(number=number,nu_energy=(e+p+difference)/2,antinu_energy=(e+p-difference)/2,electron_energy=e,positron_energy=p,FDmoments_minus=minus,FDmoments_plus=plus,tail=tail)

# A fresh gamma trace, defining all matrices locally (not importing author's spinor arrays).
S=[np.array([[0,1],[1,0]],complex),np.array([[0,-1j],[1j,0]],complex),np.diag([1.,-1.])];Z=np.zeros((2,2),complex);G=[np.diag([1.,1.,-1.,-1.])]+[np.block([[Z,s],[-s,Z]])for s in S];G5=1j*G[0]@G[1]@G[2]@G[3];metric=np.array([1.,-1.,-1.,-1.]);I=np.eye(4)
def dot(p,q):return float(np.sum(metric*p*q))
def slash(p):return sum(metric[u]*p[u]*G[u]for u in range(4))
def adj(x):return G[0]@x.conj().T@G[0]
def trace(pm,pp,k,a,v,g,m=ME):
 J=[G[u]@(I-G5)for u in range(4)];C=[G[u]@(v*I-g*G5)for u in range(4)]
 val=sum(metric[u]*metric[w]*np.trace(slash(k)@J[u]@slash(a)@adj(J[w]))*np.trace((slash(pp)-m*I)@C[u]@(slash(pm)+m*I)@adj(C[w]))for u in range(4)for w in range(4))
 return float(val.real)*GF*GF/8

def cm(s,c,m=ME):
 E=math.sqrt(s)/2;p=math.sqrt(E*E-m*m);k=E
 return [np.array(z,float)for z in [[E,0,0,p],[E,0,0,-p],[k,k*math.sqrt(1-c*c),0,k*c],[k,-k*math.sqrt(1-c*c),0,-k*c]]]
def scalar_cm(s,c,v,a,m=ME):
 beta=math.sqrt(1-4*m*m/s);A0=2*(v*v+a*a)+8*m*m/s*(v*v-a*a);A1=8*v*a*beta;A2=2*(v*v+a*a)*beta*beta
 return GF*GF*s*s/2*(A0+A1*c+A2*c*c)
