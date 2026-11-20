"""Fresh-state independent original-delta/spinor and event probes; no author edits."""
from pathlib import Path
import sys,json,hashlib,math,time,atexit
import numpy as np
from scipy.integrate import quad
from scipy.special import expit
R=Path(__file__).resolve().parents[1];H=Path(__file__).resolve().parent;sys.dont_write_bytecode=True
sys.path[:0]=[str(R/'positron_capture'),str(R/'relativistic_delta_oracle'),str(R/'relativistic_capture_oracle')]
import pc_kernel as pc,oracle,events
files=[p for p in (R/'positron_capture').iterdir()if p.is_file()];hashes={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest()for p in files};checks=[];rows=[];phase='start';start=time.monotonic()
def encode(x):
 if isinstance(x,np.ndarray):return dict(real=x.real.tolist(),imag=x.imag.tolist())if np.iscomplexobj(x)else x.tolist()
 if isinstance(x,np.generic):return x.item()
 raise TypeError(str(type(x)))
def save():
 (H/'results.json').write_text(json.dumps(dict(phase=phase,checks=checks,rows=rows,runtime=time.monotonic()-start,source_sha256=hashes,end_source_sha256={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest()for p in files}),default=encode,indent=2)+'\n')
atexit.register(save)
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and abs(error)<=tol)))
b,meta=pc.matched(1.7,2.3e-7,.27);rows.append(dict(bath=vars(b),matched=meta));phase='forward roots'
# Independent momentum integrals; no author EOS quadrature or root used to assess densities.
for name,mass,mu,target in [('n',pc.crossed.MN,b.mu_n,2.3e-7*.73),('p',pc.crossed.MP,b.mu_p,2.3e-7*.27)]:
 scale=math.sqrt(2*mass*b.T)
 val=quad(lambda x:x*x*expit((mu-math.hypot(mass,scale*x))/b.T),0,math.sqrt(160),epsabs=1e-20,epsrel=3e-11)[0]*scale**3/(math.pi**2*events.HC**3)
 gate('fresh independent '+name+' density',val/target-1,3e-9)
def nlep(sign):
 return quad(lambda x:x*x*expit((sign*b.mu_e-math.hypot(pc.ME,b.T*x))/b.T),0,100,epsabs=1e-18,epsrel=3e-11)[0]*b.T**3/(math.pi**2*events.HC**3)
ne,np_=nlep(1),nlep(-1);gate('fresh independent net charge',(ne-np_)/(2.3e-7*.27)-1,3e-9);rows.append(dict(independent_electron_density=ne,independent_positron_density=np_,pair_ratio=np_/ne));save()
for Enu,El,c in [(5.7,4.31,-.37),(3.6,2.17,.61)]:
 phase='differential'
 p=oracle.Point(b.T,pc.crossed.MP,pc.crossed.MN,b.mu_p,b.mu_n,Enu,El,c,pc.ME,-b.mu_e,True,pc.crossed.MW)
 ref=[]
 for order,tail in [(16,50.),(28,70.)]:ref.append(oracle.integrate(p,order=order,nphi=8,tail=tail))
 actual=pc.differential(b,Enu,El,c,energy_order=28,phi_order=8,tail=70.)
 for key in ['j','chi']:
  gate('independent differential refine '+key,ref[0][key]/ref[1][key]-1,3e-8);gate('fresh differential absolute '+key,actual[key]['value']/ref[1][key]-1,3e-8)
 gate('fresh shell',ref[-1]['max_shell_residual'],3e-12);gate('direct KMS',actual['j']['log']-actual['chi']['log']+(Enu-(b.mu_n-b.mu_p-b.mu_e))/b.T,3e-11)
 rows.append(dict(kind='differential',inputs=vars(p),reference=ref,response=actual));save()
# Fresh TOTAL event integral: original moving-target CM coordinate + explicit v-spinors.
eb=events.Bath(b.T,pc.crossed.MP,pc.crossed.MN,b.mu_p,b.mu_n,-b.mu_e,Enu=5.7,ml=pc.ME,anti=True,Mweak=pc.crossed.MW)
for order in [8,12]:
 phase='CM total '+str(order)
 r=events.integrate(eb,radial=order,incident=order,outgoing=2*order,azimuth=order,tail=60.);rows.append(dict(kind='CM',order=order,result=r));gate('CM fresh target density',r['density']/(2.3e-7*.27)-1,3e-8);save()
for order in [16,20]:
 if time.monotonic()-start>95:raise TimeoutError('Review total budget before new integral')
 phase='response total '+str(order)
 r=pc.integrate(b,5.7,lepton_order=order,angle_order=16,energy_order=28,phi_order=8,tail=70.,outer_tail=32.,budget_seconds=min(110.,115-(time.monotonic()-start)))
 rows.append(dict(kind='response',order=order,result=r));save()
cm=[x['result']for x in rows if x.get('kind')=='CM'];rs=[x['result']for x in rows if x.get('kind')=='response']
for key in ['j','chi']:
 gate('fresh CM convergence '+key,cm[0][key]['value']/cm[1][key]['value']-1,3e-6)
 gate('fresh total response convergence '+key,rs[0][key]['value']/rs[1][key]['value']-1,3e-6)
 gate('fresh absolute total CM response '+key,rs[1][key]['value']/cm[1][key]['value']-1,3e-6)
phase='complete';save();print(len(checks),all(x['passed']for x in checks),time.monotonic()-start);print([c for c in checks if not c['passed']])
