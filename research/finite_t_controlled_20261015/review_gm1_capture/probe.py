"""Independent adaptive DIRECT-El/cos outer quadrature, shared differential kernel."""
from pathlib import Path
import sys,json,time,math,hashlib,atexit
import numpy as np
from scipy.integrate import quad_vec
from numpy.polynomial.legendre import leggauss
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'gm1_capture_integrals'));import capture as C
start=time.monotonic();phase='start';rows=[];checks=[];partial=[];calls=0
files=[ROOT/'gm1_capture_integrals'/x for x in ['capture.py','validation_results.json','mu_q16.json','mu_q20.json','mu_q24.json','mu_cos20.json','pre_empty_domain_metadata.py','zero_field_initial8_results.json','zero_field_initial12_results.json']]+[ROOT/'gm1_differential_response/adapter.py'];hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in files}
def save():
 end={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in files}
 (HERE/'results.json').write_text(json.dumps(dict(passed=phase=='complete'and all(c['passed']for c in checks)and hashes==end,phase=phase,runtime_seconds=time.monotonic()-start,calls=calls,checks=checks,rows=rows,partial=partial,source_sha256=hashes,end_source_sha256=end),indent=2)+'\n')
atexit.register(save)
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
m=C.A.matter_state(8,.22,.17,.04);s=m.ordered();ml=C.A.leptons.ELECTRON_MASS;mul=m.electron.chemical_potential;Enu=40.;delta=s.potential_a-s.potential_b;scale=np.array([958899.1195336229,745802384.5583102]);phase='forward EOS'
for k,v in m.forward_gates.items():
 tol={'normalized_nuclear_root_residual':2e-9,'nuclear_Euler_residual_MeV_fm3':2e-8,'field_energy_decomposition_residual':1e-10,'charge_fraction_residual':2e-10,'antinucleon_ratio':1e-8}[k];gate('unperturbed '+k,v,tol)
gate('bare mass939 separate from Dirac',s.weak_bare_mass-939.,1e-12);gate('independent electron/muon neutrality',(m.electron.net_density+m.muon.net_density)/m.nuclear.nB-(m.Ye+m.Ymu),3e-10);rows.append(dict(forward=m.asdict()));save()
# Closed-form collinear support crossings, no scan or author energy panels.
edges=[(X*X+ml*ml)/(2*X)for X in [delta,2*Enu+delta]if X>0]
allbreaks=sorted(set([ml,5.,15.,40.,90.,130.,170.,200.,240.,300.,400.,600.,900.]+edges))
def run(order,upper):
 global calls,partial
 x,g=leggauss(order);sums=np.zeros(2);errsum=0.;panels=[v for v in allbreaks if ml<=v<=upper];partial=[]
 def fun(E):
  global calls
  if time.monotonic()-start>108:raise TimeoutError('Bounded independent outer quadrature')
  p=math.sqrt((E-ml)*(E+ml));w=Enu-E+delta;cut=(Enu*Enu+p*p-w*w)/(2*Enu*p);cmax=min(1.,cut)
  if cmax<=-1:return np.zeros(2)
  total=np.zeros(2)
  for c,wgt in zip(-1+(x+1)*(cmax+1)/2,g*(cmax+1)/2):
   r=C.A.differential(s,Enu,E,float(c),ml,mul,energy_order=24,phi_order=4,tail=80);calls+=1
   for j,name in enumerate(['chi','j']):
    v=r[name]['log']
    if v is not None:total[j]+=wgt*math.exp(v-math.log(scale[j]))
  return total
 for lo,hi in zip(panels[:-1],panels[1:]):
  val,err,info=quad_vec(fun,lo,hi,epsabs=2e-11,epsrel=2e-7,limit=120,full_output=True)
  if not info.success:raise ArithmeticError('Independent adaptive outer panel failed')
  sums+=val;errsum+=err;partial.append(dict(lo=lo,hi=hi,sum_scaled=sums.tolist(),error_scaled=errsum,order=order));save()
 return dict(chi=float(sums[0]*scale[0]),j=float(sums[1]*scale[1]),angular_order=order,upper=upper,estimated_scaled_adaptive_error=errsum,edges=edges)
for order,upper in [(12,600.),(20,900.)]:
 phase=f'independent {order} {upper}';r=run(order,upper);rows.append(r);save()
a,b=rows[-2:]
for name in ['chi','j']:
 gate('new independent adaptive refinement '+name,a[name]/b[name]-1,1e-6)
 target=scale[0]if name=='chi'else scale[1];gate('new adaptive vs frozen electron pilot '+name,b[name]/target-1,1e-6)
gate('separate coefficient KMS',math.log(b['j']/b['chi'])+(Enu-(mul+s.total_mu_b-s.total_mu_a))/s.T,3e-12)
phase='complete';save();print(len(checks),all(c['passed']for c in checks),time.monotonic()-start,calls)
