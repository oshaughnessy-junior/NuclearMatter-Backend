"""Bounded source, angular, streaming and boundary-ledger author controls."""
from pathlib import Path
import sys,json,math,importlib.util,time,hashlib
import numpy as np
from scipy.special import expit
from scipy.integrate import quad
HERE=Path(__file__).parent;ROOT=HERE.parent;sys.dont_write_bytecode=True
s=importlib.util.spec_from_file_location('slab_author_owned',HERE/'slab.py');B=importlib.util.module_from_spec(s);sys.modules[s.name]=B;s.loader.exec_module(B)
checks=[];start=time.monotonic()
def ck(n,error,tol):checks.append(dict(name=n,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and abs(error)<=tol)))
def dump(name,o):(HERE/name).write_text(json.dumps(o,indent=2,default=lambda x:x.item()if isinstance(x,np.generic)else str(x)))
# Independent target transverse shell with analytically averaged target azimuth.
def independent_D(E,F,c,pos):
 M=.51099895069;GF=1.1663785e-11;HBAR=6.582119569e-22;T=2.;mut=-B.MU if pos else B.MU;v=.96258;a=.5;om=E-F;Q=math.sqrt(om*om+2*E*F*(1-c));s=2*E*F*(1-c);lo=(-om+Q*math.sqrt(1+4*M*M/s))/2;hi=max(lo,mut,mut-om)+90*T;kz=(E*E+Q*Q-F*F)/(2*Q);kx2=max(0.,E*E-kz*kz);h=s/2;L,R=(v+a)**2,(v-a)**2
 if pos:L,R=R,L
 pz=lambda e:(2*e*om-s)/(2*Q)
 rmax=math.sqrt(max(0.,hi*hi-M*M-pz(hi)**2))
 def f(r):
  rr=math.sqrt(1+4*(M*M+r*r)/s);e=(-om+Q*rr)/2;de=2*Q*r/(s*rr);u=E*e-kz*pz(e);z=kx2*r*r/2;am=L*(u*u+z)+R*((u-h)**2+z)-M*M*(v*v-a*a)*h
  return de*am*expit((mut-e)/T)*expit((e+om-mut)/T)
 val=quad(f,0,rmax,epsabs=1e-8,epsrel=5e-10,limit=120)[0]
 return F/(64*math.pi**3*E*Q*HBAR)*16*GF*GF*val
# Unlike full energy+mu selfstate, chosen directional modes are finite.
edge_rows=[]
for E,F,u,v in [(5.3,12.8,-.5773502691896257,.5773502691896257),(9.1,9.1,-.5773502691896257,.5773502691896257),(4.1,8.3,.5773502691896257,.5773502691896257)]:
 a=B.angular_edge(E,F,u,v,azimuth=32);b=B.angular_edge(E,F,u,v,azimuth=64)
 def phi(ph):
  c=u*v+math.sqrt((1-u*u)*(1-v*v))*math.cos(ph)
  return sum(independent_D(E,F,c,pos)for pos in[False,True])/(2*math.pi)
 ref=quad(phi,0,2*math.pi,epsabs=2e-10,epsrel=2e-8,limit=100)[0];ck('independent directional edge '+str((E,F,u,v)),b/ref-1,3e-6);ck('directional phi32/64 '+str((E,F,u,v)),a/b-1,3e-6);edge_rows.append(dict(E=E,F=F,mu=u,mup=v,phi32=a,phi64=b,independent=ref))
grids=[]
for n in [16,32,64]:
 g=B.grid(azimuth=n,inner=32,tail=80.);grids.append(g);dump('partial_grids.json',dict(grids=grids,edges=edge_rows,checks=checks,pins=B.PINS))
g=grids[-1];E=np.array(g['E']);W=np.array(g['W']);m=np.array(g['mu']);A=np.array(g['A']);runs=[]
for k in [0,1]:
 a=np.array(grids[k]['Kbar']);b=np.array(grids[k+1]['Kbar']);mask=b>0;ck('full directional phi'+str((16*2**k,32*2**k)),np.max(abs(a[mask]/b[mask]-1)),3e-6)
ck('physical direct reverse KMS',g['kms_residual'],3e-10)
for munu in[-.7,.4]:
 f=expit((munu-E)/B.T);c=B.collision(g,f);ck('LTE physical collision '+str(munu),max(abs(c['dn']))/c['activity'],3e-12);df,ledger=B.rhs(g,np.tile(f,(3,1)),f,f);ck('LTE collision+streaming '+str(munu),max(abs(df.ravel())),3e-11)
# Fresh anisotropic state: independently construct all directed events, include both holes.
f=expit((-.7-E)/(2.2+.25*m));F=A*f[:,None]*(1-f[None,:]);dn=F.sum(axis=0)-F.sum(axis=1);c=B.collision(g,f);ck('anisotropic dense vs paired',max(abs(dn-c['dn']))/F.sum(),3e-13);ck('anisotropic number',sum(dn)/F.sum(),3e-14);ck('anisotropic energy sign',(-E@dn-c['heat'])/max(abs(c['heat']),1e-300),3e-13)
macro=-np.log(f/(1-f))@dn+c['heat']/B.T;ck('anisotropic entropy macro/micro',(macro-c['entropy'])/F.sum(),3e-12);ck('anisotropic entropy nonnegative',min(c['entropy'],0)/F.sum(),1e-15)
# Exact solution of the declared semi-discrete upwind streaming chain, NOT continuum.
zero=B.evolve(g,collisions=False);ff=np.array(zero['final']);f0=np.array(zero['initial']);left,right=B.boundary(E);exact=np.zeros_like(ff);nc=zero['cells'];t=zero['duration'];cross=zero['crossing_time']
for z in range(nc):
 for j in range(len(E)):
  dist=z if m[j]>0 else nc-1-z;boundary=left[j]if m[j]>0 else right[j];at=abs(m[j])*nc/cross*t;poly=sum(at**r/math.factorial(r)for r in range(dist+1));exact[z,j]=boundary+(f0[z,j]-boundary)*math.exp(-at)*poly
ck('zero collision exact upwind streaming',max(abs(ff.ravel()-exact.ravel())),3e-11)
for rtol,h in [(2e-9,.004),(2e-11,.001)]:
 r=B.evolve(g,rtol=rtol,max_step=h);runs.append(r)
 scaleN=r['boundary_in_number']+r['boundary_out_number'];scaleE=r['boundary_in_energy']+r['boundary_out_energy']+abs(r['thermostat_heat']);ck('boundary number ledger '+str(rtol),r['number_ledger']/scaleN,3e-12);ck('boundary energy+thermostat ledger '+str(rtol),r['energy_ledger']/scaleE,3e-12);ck('interior occupations '+str(rtol),float(not 0<r['min_occupation']<=r['max_occupation']<1),0);ck('positive collision entropy '+str(rtol),min(r['collision_entropy'],0),1e-30)
a,b=runs;ck('nontrivial collision changes spectrum',float(np.max(abs(np.array(b['final'])-np.array(zero['final'])))<.001),0);ck('time refined occupation',np.max(abs(np.array(a['final'])-np.array(b['final']))),3e-8);ck('time refined thermostat heat',a['thermostat_heat']/b['thermostat_heat']-1,3e-7)
# Edge-rule refinement on final full physical slab observable, not KMS alone.
coarse=B.evolve(grids[1],rtol=2e-11,max_step=.001);ck('phi32/64 slab heat',coarse['thermostat_heat']/b['thermostat_heat']-1,3e-6);ck('phi32/64 slab spectrum',np.max(abs(np.array(coarse['final'])-np.array(b['final']))),3e-7)
for n,fn in [('selfmode',lambda:B.angular_edge(3.,3.,.2,.2)),('nonfinite',lambda:B.angular_edge(3.,float('nan'),.2,-.2)),('badphi',lambda:B.angular_edge(3.,4.,.2,-.2,azimuth=True))]:
 try:fn();bad=True
 except ValueError:bad=False
 ck('reject '+n,float(bad),0)
r=dict(checks=checks,passed=sum(x['passed']for x in checks),failed=[x for x in checks if not x['passed']],edges=edge_rows,grids=grids,zero_collision=zero,runs=runs,phi32_run=coarse,pins=B.PINS,seconds=time.monotonic()-start,scope='finite energy S2 three-cell upwind slab; fixed pair thermostat; nue only; no angular/spatial/energy continuum certificate')
dump('results.json',r);print(json.dumps(dict(passed=r['passed'],failed=r['failed'],seconds=r['seconds'],final_heat=b['thermostat_heat'],final_dE=b['radiation_energy_change'],net_boundary_E=b['boundary_in_energy']-b['boundary_out_energy']),indent=2))
