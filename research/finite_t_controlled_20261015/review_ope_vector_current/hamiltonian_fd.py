"""Finite external-field derivative of Fourier-resolved static Hamiltonian.

Separate program: no probe/author-driver import and no Ward defect used.
"""
from pathlib import Path
from itertools import product
import importlib.util,sys,json,math,time
import numpy as np
HERE=Path(__file__).resolve().parent;sys.dont_write_bytecode=True
sp=importlib.util.spec_from_file_location('static_H_review',HERE.parent/'ope_vector_current/current.py');C=importlib.util.module_from_spec(sp);sys.modules[sp.name]=C;sp.loader.exec_module(C)
ss=[np.array([[0,1],[1,0]],complex),np.array([[0,-1j],[1j,0]]),np.diag([1.,-1.])];basis=list(product(range(2),repeat=4));perm=[(2*s1+s2)*4+2*t1+t2 for s1,t1,s2,t2 in basis]
def lift(line,s,t):
 z=np.zeros((16,16),complex)
 for j,a in enumerate(basis):
  for k,b in enumerate(basis):
   if a[2*(1-line):2*(1-line)+2]==b[2*(1-line):2*(1-line)+2]:z[j,k]=s[a[2*line],b[2*line]]*t[a[2*line+1],b[2*line+1]]
 return z
def eps(a,b,c):return 0 if len({a,b,c})<3 else (1 if(a,b,c)in[(0,1,2),(1,2,0),(2,0,1)]else -1)
k1=np.array([20.,0,0]);k2=np.array([0,30.,0]);ks=[k1,k2];g=1.01/140.;mp=140.
field=[[ -1j*g*lift(l,sum(ks[l][i]*ss[i]for i in range(3)),ss[a])/(ks[l]@ks[l]+mp*mp)for a in range(3)]for l in range(2)]
source=[[[lift(l,ss[i],ss[a])for i in range(3)]for a in range(3)]for l in range(2)]
def energy(eta,cart_b,spatial_i):
 H=np.zeros((16,16),complex)
 for ix,iy in product(range(8),repeat=2):
  x,y=2*math.pi*ix/8,2*math.pi*iy/8
  phase=[np.exp(-2j*x),np.exp(-3j*y)];external=eta*np.exp(1j*(2*x+3*y))
  phi=[sum(phase[l]*field[l][a]for l in range(2))for a in range(3)]
  for a in range(3):
   H+=mp*mp/2*phi[a]@phi[a]
   for i in range(3):
    dphi=sum(-1j*ks[l][i]*phase[l]*field[l][a]for l in range(2))
    if i==spatial_i:dphi+=sum(eps(a,cart_b,c)*external*phi[c]for c in range(3))
    S=sum(phase[l]*source[l][a][i]for l in range(2))
    H+=.5*dphi@dphi+g*S@dphi
 return H/64
checks=[];rows=[];start=time.monotonic()
for raising in [False,True]:
 current=C.direct(k1,k2,raising=raising)
 for step in [.017,.041]:
  fd=[]
  for i in range(3):
   cart=[]
   for b in [0,1]:cart.append(-(energy(step,b,i)-energy(-step,b,i))/(2*step))
   fd.append(cart[0]+(1j if raising else -1j)*cart[1])
  fd=np.array(fd);expected=current['total'][:,perm,:][:,:,perm]
  error=float(np.max(abs(fd-expected))/np.max(abs(expected)))
  checks.append(dict(name=str((raising,step))+' explicit H[v] central difference',error=error,tolerance=3e-10,passed=error<=3e-10))
  rows.append(dict(raising=raising,step=step,error=error))
out=dict(checks=checks,passed=sum(x['passed']for x in checks),failed=[x for x in checks if not x['passed']],rows=rows,seconds=time.monotonic()-start,method='2D periodic Fourier field, stationary pion held fixed for first variation; actual Hamiltonian quadratics and source term, no current formula/defect in H evaluator')
(HERE/'hamiltonian_fd_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k]for k in ['passed','failed','seconds']}))
