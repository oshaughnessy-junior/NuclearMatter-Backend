"""Independent overlap projections, all norms/orders and time-error accounting."""
from pathlib import Path
import sys,json,math,hashlib
import numpy as np
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parents[1];H=Path(__file__).parent;A=R/'conductive_spatial_refinement'
sys.path.insert(0,str(R/'conductive_thermal_slab'));import slab
p=slab.Physics();checks=[];rows=[]
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
manifest=json.loads((A/'MANIFEST.json').read_text())
for group,root in [('files',A),('upstream',R)]:
 for name,sha in manifest[group].items():gate('frozen '+name,int(hashlib.sha256((root/name).read_bytes()).hexdigest()!=sha),0)
d=json.loads((A/'results.json').read_text());t=json.loads((A/'time_separation_results.json').read_text());coeff=json.loads((A/'coefficient_range.json').read_text())
gate('time supplement pins original report',int(t['original_results_sha256']!=hashlib.sha256((A/'results.json').read_bytes()).hexdigest()),0)
def project(v,N):
 v=np.array(v);M=len(v)
 return np.array([sum(z*max(0,min((i+1)/N,(j+1)/M)-max(i/N,j/M))for j,z in enumerate(v))*N for i in range(N)])
def norms(v):
 v=np.array(v);return dict(L1=float(sum(abs(v))/len(v)),L2=float(math.hypot(*v)/math.sqrt(len(v))),Linf=float(max(abs(v))))
runs={(z['run']['controls']['N'],z['run']['controls']['steps']):z['run']for report in [d,t]for z in report['rows']if'run'in z}
for key,r in runs.items():
 N,steps=key;W=[p.W(z)for z in r['theta']];S=float(np.mean([p.entropy(z)for z in r['theta']]))
 gate('endpoint actual EOS '+str(key),max(abs(np.array(W)-r['final_W'])),3e-12)
 gate('endpoint energy ledger '+str(key),float(np.mean(W))-r['initial_energy'],5e-10)
 gate('endpoint entropy monotonic '+str(key),min(0,S-r['initial_entropy']),1e-12)
 gate('zero energy/entropy boundary flux '+str(key),max(abs(h[n])for h in r['history']for n in ['boundary_energy_in','boundary_energy_out','boundary_entropy']),0)
space=[]
for N in [4,8]:
 dif=np.array(runs[(N,16)]['final_W'])-project(runs[(2*N,16)]['final_W'],N);norm=norms(dif);space.append(norm)
 raw=next(z['spatial_comparison']['error']for z in d['rows']if'spatial_comparison'in z and z['spatial_comparison']['coarse_N']==N)
 for name in norm:gate('independent spatial '+str(N)+' '+name,norm[name]/raw[name]-1,3e-12)
 rows.append(dict(independent_spatial=dict(N=N,norms=norm)))
for name in space[0]:
 order=math.log2(space[0][name]/space[1][name]);raw=next(z['spatial_rate']['measured_order']for z in d['rows']if'spatial_rate'in z and z['spatial_rate']['norm']==name)
 gate('independent measured order '+name,order-raw,3e-12)
 rows.append(dict(measured_order=dict(norm=name,value=order)))
dt1=np.array(runs[(16,16)]['final_W'])-np.array(runs[(16,32)]['final_W']);dt2=np.array(runs[(16,32)]['final_W'])-np.array(runs[(16,64)]['final_W'])
for name in space[0]:
 ratio=norms(dt2)[name]/norms(dt1)[name]
 gate('independent time ratio '+name,ratio-.5,.03)
 for N in [4,8]:
  estimate=2*norms(project(dt1,N))[name];frac=estimate/space[0 if N==4 else 1][name]
  raw=next(z['time_vs_space']['fraction']for z in d['rows']if'time_vs_space'in z and z['time_vs_space']['compare_grid']==N and z['time_vs_space']['norm']==name)
  gate('independent Richardson estimate '+str(N)+' '+name,frac-raw,3e-12)
  rows.append(dict(time_fraction=dict(N=N,norm=name,value=frac)))
for N in [4,8]:
 a=np.array(runs[(N,16)]['final_W'])-project(runs[(2*N,16)]['final_W'],N)
 b=np.array(runs[(N,32)]['final_W'])-project(runs[(2*N,32)]['final_W'],N)
 old,new,change=norms(a),norms(b),norms(a-b)
 raw=next(z['time_separation']['fraction']for z in t['rows']if'time_separation'in z and z['time_separation']['N']==N)
 for name in old:gate('new coarse timehalve spatial change '+str(N)+' '+name,change[name]/old[name]-raw[name],3e-12)
 rows.append(dict(space_difference_time_halving=dict(N=N,old=old,new=new,fraction={name:change[name]/old[name]for name in old})))
gate('fresh forward particle density',p.thermo(p.T0).gas['density']/5.3e-8-1,3e-10)
for r in coeff['rows']:
 th=r['theta'];k=p.k(th);cv=p.cv(th)
 gate('fresh actual conductivity '+str(th),k/r['kappa_ratio']-1,2e-14)
 gate('fresh actual Cv '+str(th),cv/r['Cv_ratio']-1,2e-14)
 gate('fresh actual diffusivity '+str(th),(k/cv)/r['diffusivity_ratio']-1,2e-14)
rows.append(dict(diffusivity_span=max(r['diffusivity_ratio']for r in coeff['rows'])-min(r['diffusivity_ratio']for r in coeff['rows'])))
(H/'audit.json').write_text(json.dumps(dict(checks=checks,rows=rows,all_pass=all(c['passed']for c in checks)),indent=2)+'\n')
print(json.dumps(dict(checks=len(checks),failed=[c for c in checks if not c['passed']])))
