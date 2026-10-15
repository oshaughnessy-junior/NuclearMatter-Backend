"""Resolve independent cold quadrature and probe LTE, KMS and overflow locus."""
# Import review helpers without replaying the initial experiment.
from pathlib import Path
import json,math,time
p=Path(__file__).resolve().parent
text=(p/'probe.py').read_text();exec(text.split("for label,T,n,E,rt,mu in")[0])
initial=json.loads((P/'results.json').read_text());original_save=save
# Separate output; never overwrite preserved initial failures.
def save():
 (P/'refined_results.json').write_text(json.dumps(dict(checks=checks,rows=rows,failures=failures,source_hashes_before=before,source_hashes_after=sha(),runtime_seconds=time.monotonic()-start),indent=2)+'\n')
sp=r.nc.state_density(.2,.0042,0.);E=19.6;rad=r.Radiation('FD',.7,5.1);cut=r.upper(sp,E,70.)
ref=np.array(next(x for x in initial['rows'] if x['label']=='heldout_cold')['author_q96']['values'])
vals=[]
for qo,eo in [(96,48),(144,64)]:
 val=independent(sp,E,rad,cut,qo,eo);vals.append(val);rows.append(dict(label=f'cold q{qo} E{eo}',values=val.tolist()))
 for i,name in enumerate(r.LABELS):gate(f'cold refined{qo} authorcomparison '+name,abs(val[i]/ref[i]-1),3e-5)
 save()
for i,name in enumerate(r.LABELS):gate('cold independent96->144 '+name,abs(vals[0][i]/vals[1][i]-1),3e-5)
# Detailed balance checked with two actual response evaluations, not exp imposed.
for T,n,Ein in [(1.1,.0017,13.7),(.2,.0042,19.6),(.08,.0042,19.6)]:
 sp=r.nc.state_density(T,n,0.);pair=r.nc.pair_from_species(sp)
 for Ep,c in [(Ein*.91,-.63),(Ein*1.07,.27),(Ein,.999)]:
  lf=r.nc.log_kernel(pair,Ein,Ep,c);lr=r.nc.log_kernel(pair,Ep,Ein,c)
  gate(f'actualreverse DOS T{T}Ep{Ep}c{c}',abs(lr-lf-2*math.log(Ein/Ep)+(Ein-Ep)/T),2e-11)
 # Compare final blocking at fixed kinematics independently, including no extra DOS.
 rad=r.Radiation('FD',T*1.7,3.2);Ep=Ein*.97;v=r.integrand(sp,Ein,Ep,rad,angle_order=144,coordinate='cos');cs,ws=leggauss(144)
 bare=sum(w*r.nc.kernel(pair,Ein,Ep,float(c)) for c,w in zip(cs,ws));fi=rad.f(Ein);fo=rad.f(Ep)
 gate(f'finalblocking T{T}',abs(v[0]/(bare*(1-fo))-1),3e-12)
 gate(f'incomingloss T{T}',abs(v[5]/(bare*fi*(1-fo))-1),3e-12)
# Exact LTE actualreverse integrals: independent reverse kernels, not analytic zero.
sp=r.nc.state_density(1.1,.0017,0.);E=13.7;rad=r.Radiation('FD',sp.T,3.2);val=independent(sp,E,rad,r.upper(sp,E,70.),56,32);rows.append(dict(label='LTE actualreverse',values=val.tolist()))
gate('LTE actualreverse grossscaled residual',abs(val[6])/(val[4]+val[5]),3e-12)
# Pin exact broader-domain failure location, preserving log quantities and response.
sp=r.nc.state_density(.08,.0042,0.);E=19.6;rad=r.Radiation('FD',.4,5.);Ep=E+720*sp.T
try:r.integrand(sp,E,Ep,rad,angle_order=48);failure=False
except OverflowError as exc:failure=True;rows.append(dict(label='cold overflow exact integrand',T=sp.T,Ein=E,Eout=Ep,exponent=(Ep-E)/sp.T,exception=repr(exc)))
gate('preserved overflow reproduces at exponent720',float(not failure),.1)
# Stable independent integral at same cold bath: no division or ordinary reverse exp.
for qo,eo in [(96,48),(144,64)]:
 val=independent(sp,E,rad,r.upper(sp,E,70.),qo,eo);rows.append(dict(label=f'cold08 independent{qo}',values=val.tolist()))
 if qo==96:first=val
 else:
  for i,name in enumerate(r.LABELS):gate('cold08 independent96->144 '+name,abs(first[i]/val[i]-1),3e-5)
gate('source/data hashes unchanged',float(before!=sha()),.1);save();print('checks',sum(c['passed']for c in checks),'/',len(checks),'runtime',time.monotonic()-start)
for c in checks:
 if not c['passed']:print('FAIL',c)
