from pathlib import Path
import sys,time,json,math,hashlib,atexit
import numpy as np
sys.dont_write_bytecode=True
from oracle import *
HERE=Path(__file__).resolve().parent;start=time.monotonic();checks=[];rows=[];phase='start'
def encode(v):
 if isinstance(v,np.ndarray):return {'real':v.real.tolist(),'imag':v.imag.tolist()}if np.iscomplexobj(v)else v.tolist()
 if isinstance(v,np.generic):return v.item()
 raise TypeError(type(v).__name__)
def save():
 (HERE/'results.json').write_text(json.dumps(dict(phase=phase,passed=phase=='complete'and all(c['passed']for c in checks),runtime_seconds=time.monotonic()-start,checks=checks,rows=rows,source_sha256=hashlib.sha256((HERE/'oracle.py').read_bytes()).hexdigest()),default=encode,indent=2)+'\n')
atexit.register(save)
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
for i,p in enumerate(POINTS):
 phase=f'point{i}';vals=[]
 for order,nphi,tail in [(12,4,30),(24,8,40),(32,16,55)]:
  if time.monotonic()-start>100:raise TimeoutError('Bounded experiment guard')
  r=integrate(p,order,nphi,tail);vals.append(r)
 row=dict(point=vars(p),runs=vals);rows.append(row);a,b,c=vals
 for label,z in [('coarse',a),('medium',b)]:
  for field in ['chi','j']:
   gate(f'{i} {label} {field} refinement',z[field]/c[field]-1,3e-7)
  gate(f'{i} {label} full W refinement',np.linalg.norm(z['W']-c['W'])/np.linalg.norm(c['W']),3e-7)
 ratio=math.exp(-(p.Enu-(p.mul+p.mub-p.mua))/p.T)
 gate(f'{i} separately integrated emission KMS',c['j']/c['chi']/ratio-1,2e-12)
 gate(f'{i} delta mass shell',c['max_shell_residual'],2e-14)
 for j in range(2):
  gate(f'{i} W{j} Hermiticity',np.linalg.norm(c['W'][j]-c['W'][j].conj().T)/np.linalg.norm(c['W'][j]),1e-14)
  gate(f'{i} W{j} PSD',min(0,np.linalg.eigvalsh(c['W'][j]).min())/np.linalg.norm(c['W'][j]),1e-13)
 # Original delta derivative using an independent symmetric pz finite difference.
 pa,pb,J=shell(p,c['umax']*.17,.37);z=pa[3];r2=pa[1]**2+pa[2]**2;h=1e-3
 def delta(z):return math.sqrt(r2+z*z+p.ma*p.ma)+c['w']-math.sqrt(r2+(z+c['q'])**2+p.mb*p.mb)
 gate(f'{i} independent delta derivative',abs((delta(z+h)-delta(z-h))/(2*h))/J-1,2e-8)
 save()
phase='complete';save();print(len(checks),all(c['passed']for c in checks),time.monotonic()-start)
