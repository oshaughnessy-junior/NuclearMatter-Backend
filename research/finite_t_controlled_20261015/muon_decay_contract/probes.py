from pathlib import Path
import sys,time,json,math,hashlib,atexit
from scipy.special import log_expit
sys.dont_write_bytecode=True
from decay import *
HERE=Path(__file__).resolve().parent;start=time.monotonic();rows=[];checks=[];phase='start'
def save():(HERE/'results.json').write_text(json.dumps(dict(passed=phase=='complete'and all(c['passed']for c in checks),phase=phase,checks=checks,rows=rows,runtime_seconds=time.monotonic()-start,source_sha256=hashlib.sha256((HERE/'decay.py').read_bytes()).hexdigest()),indent=2,default=lambda x:x.tolist()if isinstance(x,np.ndarray)else x.item())+'\n')
atexit.register(save)
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
for me in [ME,0.,.9*MMU]:
 vals=[]
 for order in [16,32,48]:
  phase=f'{me}{order}';a=pair_integral(order,me=me,method='invariant');b=dalitz_integral(order,me=me,method='spinor');vals.append([a,b]);rows.append(dict(me=me,order=order,pair=a,dalitz=b));save()
  for field in ['rate','energy_e','energy_anti_e','energy_nu_mu']:gate(f'{me} {order} independent phase/matrix {field}',a[field]/b[field]-1,3e-7)
  gate(f'{me} {order} pair fourmomentum closure',a['fourmomentum_residual'],3e-13)
  gate(f'{me} {order} absolute exact-mass analytic',b['rate']/analytic_rate(me=me)-1,3e-9)
  gate(f'{me} {order} full energy ledger',(b['energy_e']+b['energy_anti_e']+b['energy_nu_mu'])/(MMU*b['rate'])-1,3e-13)
 for j in [0,1]:
  for field in ['rate','energy_e','energy_anti_e','energy_nu_mu']:gate(f'{me} refinement method{j} {field}',vals[1][j][field]/vals[2][j][field]-1,3e-9)
 if me==0:
  a=vals[-1][1]
  for key,fraction in [('energy_e',.35),('energy_anti_e',.30),('energy_nu_mu',.35)]:gate('massless flavor moment '+key,a[key]/(a['rate']*MMU*fraction)-1,3e-12)
# Spinor vs invariant and reciprocal with independently chosen non-grid physical events.
e=np.array([3.7,24.1,46.3]);pe=np.sqrt(e*e-ME*ME);lo=(MMU-e-pe)/2;hi=(MMU-e+pe)/2;ae=lo+(hi-lo)*np.array([.17,.43,.71]);pm,l,k,n=dalitz_events(e,ae,MMU,ME);spin=matrix_squared(pm,l,k,n);scalar=matrix_squared(pm,l,k,n,method='invariant');inv=matrix_squared(pm,l,k,n,inverse=True);gate('new direct spinors vs invariant',np.max(abs(spin/scalar-1)),3e-13);gate('new reciprocal spinors',np.max(abs(inv/spin-1)),3e-13)
for E in [5.,19.,43.]:
 s=spectrum(E,me=0);x=2*E/MMU;michel=4*analytic_rate(me=0)/MMU*x*x*(3-2*x);gate('massless Michel spectrum '+str(E),s/michel-1,3e-12)
# Full forward/backward occupation product; partner neutrino not discarded.
for T,mum,mue,mua,mun in [(3.,96.,29.,-4.,63.),(.7,104.,17.,-2.,89.)]:
 Em,Ee,Ea,En=(p[:,0]for p in[pm,l,k,n]);F=log_expit((mum-Em)/T)+log_expit((Ee-mue)/T)+log_expit((Ea-mua)/T)+log_expit((En-mun)/T);B=log_expit((Em-mum)/T)+log_expit((mue-Ee)/T)+log_expit((mua-Ea)/T)+log_expit((mun-En)/T);aff=mum-mue-mua-mun;gate('full four-species affinity '+str(T),np.max(abs(F-B-aff/T)),2e-12)
 rows.append(dict(thermal_product_probe=dict(T=T,mumu=mum,mue=mue,muanti_e=mua,munu_mu=mun,affinity=aff,log_forward=F,log_inverse=B)))
phase='complete';save();print(len(checks),all(c['passed']for c in checks),time.monotonic()-start)
