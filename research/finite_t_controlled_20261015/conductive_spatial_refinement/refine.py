"""Compatible cell-averaged energy spatial gate, actual nonlinear slab physics."""
from pathlib import Path
import sys,math,time,json,hashlib
import numpy as np
from numpy.polynomial.legendre import leggauss
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;R=H.parent
sys.path.insert(0,str(R/'conductive_thermal_slab'));import slab
start=time.monotonic();checks=[];rows=[]
source_paths=['conductive_thermal_slab/slab.py','screened_coulomb_transport/lorentz.py','relativistic_free_eos/free_baryons.py','thermodynamics/lepton_reference.py']
def hashes():return {p:hashlib.sha256((R/p).read_bytes()).hexdigest()for p in source_paths}
initial_hashes=hashes();p=slab.Physics();duration=.0015;amplitude=.04
(H/'PROTOCOL.json').write_text(json.dumps(dict(grids=[4,8,16],common_time_steps=16,fine_time_ladder=[16,32,64],duration=duration,initial_temperature='theta(x)=1+.04*cos(pi*x)',cell_initialization='actual EOS W averaged by Gauss; exact projection from16 to8/4',comparison='cell-averaged W, conservative pair projection; L1/L2/Linf',temporal_estimate='first-order 2*D16vs32, verified32vs64 ladder; target<.1 of8vs16 spatial difference',gates=dict(initial_quadrature=3e-11,projection=5e-11,energy=5e-10,temporal_fraction=.1),frozen_sources=initial_hashes),indent=2)+'\n')
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and abs(error)<=tol)))
def save(phase):
 (H/'results.json').write_text(json.dumps(dict(phase=phase,passed=phase=='complete'and all(c['passed']for c in checks),checks=checks,rows=rows,initial_hashes=initial_hashes,final_hashes=hashes(),runtime_seconds=time.monotonic()-start),indent=2)+'\n')
def average(N,order):
 x,w=leggauss(order);out=[]
 for i in range(N):
  locations=(i+(x+1)/2)/N;out.append(float(sum(ww*p.W(1+amplitude*math.cos(math.pi*xx)) for xx,ww in zip(locations,w)))/2)
 return np.array(out)
def project(values,N):
 a=np.asarray(values)
 if len(a)%N:raise ValueError('Exact nested conservative projection required')
 return a.reshape(N,len(a)//N).mean(axis=1)
def norms(values):
 a=np.asarray(values);return dict(L1=float(np.mean(abs(a))),L2=float(math.sqrt(np.mean(a*a))),Linf=float(np.max(abs(a))))
# New quadrature, no conductivity interpolation or sampled-T initialization.
a8=average(16,8);a12=average(16,12);gate('initial cell energy quadrature8vs12',max(abs(a8-a12)),3e-11)
initial_W={N:project(a12,N)for N in[4,8,16]}
for N in[4,8]:
 independently_averaged=average(N,12);gate('initial projection vs independent average '+str(N),max(abs(initial_W[N]-independently_averaged)),3e-11)
for N in[4,8,16]:gate('initial exact volume energy '+str(N),float(np.mean(initial_W[N])-np.mean(a12)),3e-17)
rows.append(dict(initialization=dict(fine_W8=a8.tolist(),fine_W12=a12.tolist(),compatible_W={str(N):a.tolist()for N,a in initial_W.items()})));save('initial averages qualified')

def run(N,steps):
 theta=np.array([p.inverse(W,1+amplitude*math.cos(math.pi*(i+.5)/N),(.95,1.05))for i,W in enumerate(initial_W[N])]);actual_initial=np.array([p.W(t)for t in theta]);gate('initial EOS inversion '+str((N,steps)),max(abs(actual_initial-initial_W[N])),5e-11)
 experiment_start=time.monotonic();r=slab.evolve(p,theta,duration,steps,budget_seconds=90.);r['experiment_runtime_seconds']=time.monotonic()-experiment_start
 final=np.array([p.W(t)for t in r['theta']]);r['final_W']=final.tolist();r['target_initial_W']=initial_W[N].tolist()
 ledger=max(abs(h['material_energy']-r['initial_energy'])for h in r['history']);gate('isolated actual EOS energy '+str((N,steps)),ledger,5e-10)
 gate('initial volume energy inversion '+str((N,steps)),r['initial_energy']-float(np.mean(a12)),5e-11)
 production=[h['entropy']-r['initial_entropy']for h in r['history']];gate('nonnegative actual entropy '+str((N,steps)),min(0.,min(production)),1e-12)
 gate('temperature max principle '+str((N,steps)),float(min(r['theta'])<min(theta)-1e-9 or max(r['theta'])>max(theta)+1e-9),0.)
 gate('conservative EOS inversion residual '+str((N,steps)),r['max_inversion_residual'],5e-11)
 gate('bounded experiment seconds '+str((N,steps)),max(0.,r['experiment_runtime_seconds']-120.),0.)
 rows.append(dict(run=r));save('run checkpoint '+str((N,steps)));return r
runs={}
for N in[4,8,16]:runs[(N,16)]=run(N,16)
for steps in[32,64]:runs[(16,steps)]=run(16,steps)
spatial=[]
for N in[4,8]:
 coarse=np.array(runs[(N,16)]['final_W']);fine=project(runs[(2*N,16)]['final_W'],N);diff=norms(coarse-fine);spatial.append(diff);rows.append(dict(spatial_comparison=dict(coarse_N=N,fine_N=2*N,common_steps=16,conservative_projection=True,error=diff)))
for norm in['L1','L2','Linf']:
 ratio=spatial[1][norm]/spatial[0][norm];order=math.log(1/ratio,2);gate('measured nonlinear spatial order '+norm,order-2.,.4)
 rows.append(dict(spatial_rate=dict(norm=norm,ratio=ratio,measured_order=order,scope='selected3grids only, not universal continuum proof')))
time_diffs=[]
for s1,s2 in[(16,32),(32,64)]:
 diff=norms(np.array(runs[(16,s1)]['final_W'])-np.array(runs[(16,s2)]['final_W']));time_diffs.append(diff);rows.append(dict(temporal_comparison=dict(N=16,coarse_steps=s1,fine_steps=s2,error=diff)))
for norm in['L1','L2','Linf']:
 ratio=time_diffs[1][norm]/time_diffs[0][norm];gate('first-order time ladder '+norm,ratio-.5,.03)
 # The conservative spatial comparisons are oncoarse grids: project the
 # fine-grid time difference onto each same grid before bounding contamination.
 for N in[4,8]:
  estimated=2*norms(project(np.array(runs[(16,16)]['final_W'])-np.array(runs[(16,32)]['final_W']),N))[norm]
  space=spatial[0 if N==4 else 1][norm];fraction=estimated/space
  gate('time below spatial '+str((N,norm)),max(0.,fraction-.1),0.)
  rows.append(dict(time_vs_space=dict(compare_grid=N,norm=norm,first_order_estimate=estimated,spatial_difference=space,fraction=fraction)))
# Full selectedEOS/control identity and source immutability, no cache changes.
gate('frozen source hashes unchanged',float(hashes()!=initial_hashes),0.)
rows.append(dict(cost=dict(EOS=p.eos_calls,transport=p.transport_calls),scales=p.scales()))
save('complete');print(json.dumps(dict(passed=all(c['passed']for c in checks),checks=len(checks),failures=[c for c in checks if not c['passed']],runtime=time.monotonic()-start)))
