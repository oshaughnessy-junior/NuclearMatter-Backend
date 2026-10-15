"""Bounded normalization/phase-space checks; no warm-medium claim."""
from pathlib import Path
import json,math,time,hashlib
import numpy as np
import vacuum_cc as V
P=Path(__file__).resolve().parent;start=time.monotonic();checks=[];rows=[]
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and abs(error)<=tol)))
def relative(a,b):return (a-b)/b

# Clifford algebra independent of the scattering fixtures.
for a in range(4):
 for b in range(4):gate(f'Clifford {a}{b}',np.max(abs(V.G[a]@V.G[b]+V.G[b]@V.G[a]-2*(V.METRIC[a]if a==b else 0)*V.I)),1e-14)
gate('gamma5 squared',np.max(abs(V.G5@V.G5-V.I)),1e-14)

for anti in (False,True):
 for ml,label in [(V.ME,'e'),(V.MMU,'mu')]:
  ch=V.Channel(V.MP if anti else V.MN,V.MN if anti else V.MP,ml,anti)
  threshold=ch.threshold
  threshold_distance=((ch.initial_mass+2*threshold)*ch.initial_mass-(ch.final_mass+ml)**2)/ch.initial_mass**2
  gate(f'threshold invariant {anti}{label}',threshold_distance if threshold else min(threshold_distance,0.),4e-16)
  if threshold:
   for factor in (.999999,1.):gate(f'forbidden {anti}{label}{factor}',V.total(ch,threshold*factor),0.)
   gate(f'positive just above {anti}{label}',0. if V.total(ch,threshold*(1+1e-8))>0 else 1.,0.)
   first=V.total(ch,threshold+1e-5)/math.sqrt(1e-5)
   second=V.total(ch,threshold+1e-7)/math.sqrt(1e-7)
   gate(f'square root threshold {anti}{label}',relative(first,second),3e-4)
  energies=[max(threshold+1.,10.),max(threshold+20.,50.),150.,300.]
  for E in energies:
   for cosine in (-.97,-.3,.41,.99):
    p=V.momenta(ch,E,cosine)
    amp=V.trace_amplitude(ch,p);other=V.invariant_amplitude(ch,p)
    gate(f'trace invariant anti{anti}{label}E{E}c{cosine}',relative(amp,other),2e-12)
    gate(f'momentum conservation {anti}{label}{E}{cosine}',np.max(abs(p['k']+p['pa']-p['l']-p['pb']))/math.sqrt(p['s']),2e-15)
    for key,m in [('k',0.),('pa',ch.initial_mass),('l',ml),('pb',ch.final_mass)]:
     gate(f'mass shell {key}{anti}{label}{E}{cosine}',(V.dot(p[key],p[key])-m*m)/p['s'],2e-15)
   sig8=V.total(ch,E,order=8);sig24=V.total(ch,E,order=24)
   gate(f'angular refinement {anti}{label}{E}',relative(sig8,sig24),2e-12)
   rows.append(dict(antineutrino=anti,lepton=label,E_MeV=E,threshold_MeV=threshold,sigma_cm2=sig24,point_current_static_limit_cm2=V.static_total(ch,E)))

# Independently implement Strumia/Vissani Eq6 with Delta=0, f2=g2=0.
# Their M² is not our physical amplitude²: multiply by8 GF² Vud².
for anti in(False,True):
 ch=V.Channel(939.,939.,V.MMU,anti)
 for E,cosine in [(120.,-.7),(150.,.3),(300.,.8)]:
  p=V.momenta(ch,E,cosine);q=p['k']-p['l'];t=V.dot(q,q);u=V.dot(p['k']-p['pb'],p['k']-p['pb']);M=ch.initial_mass;m=ch.lepton_mass
  A=(t-m*m)/4*(ch.gv**2*(4*M*M+t+m*m)+ch.ga**2*(-4*M*M+t+m*m))
  B=-t*ch.gv*ch.ga;C=(ch.gv**2+ch.ga**2)/4
  independent=8*V.GF**2*V.VUD**2*(A+(-1 if anti else 1)*(p['s']-u)*B+(p['s']-u)**2*C)
  gate(f'Strumia point-current symmetricM anti{anti}E{E}',relative(V.trace_amplitude(ch,p),independent),2e-12)

heavy=[]
for anti in(False,True):
 for ml,E,label in [(V.ME,20.,'e'),(V.MMU,150.,'mu')]:
  errors=[]
  for M in(1e6,1e7):
   ch=V.Channel(M,M,ml,anti);value=V.total(ch,E,order=12);static=V.static_total(ch,E)
   errors.append(abs(relative(value,static)))
   gate(f'heavy static total{anti}{label}M{M}',relative(value,static),1e-3 if M==1e6 else 1e-4)
   for cosine in(-.8,0.,.7):
    beta=math.sqrt(1-ml*ml/(E*E));expected=V.GF**2*V.VUD**2/(2*math.pi)*E*E*beta*(ch.gv**2+3*ch.ga**2+(ch.gv**2-ch.ga**2)*beta*cosine)*V.HBARC**2*1e-26
    gate(f'heavy static angle{anti}{label}M{M}c{cosine}',relative(V.differential(ch,E,cosine),expected),1e-3 if M==1e6 else 1e-4)
   heavy.append(dict(antineutrino=anti,lepton=label,M=M,E=E,sigma_cm2=value,static_cm2=static,relative_error=errors[-1]))
  gate(f'heavy limit convergence{anti}{label}',0. if errors[1]<errors[0]/8 else 1.,0.)

# m_l->0 at fixed finite M, independently parameterized (no global mass edit).
for anti in(False,True):
 ch0=V.Channel(939.,939.,0.,anti);chsmall=V.Channel(939.,939.,1e-5,anti)
 gate(f'massless lepton limit{anti}',relative(V.total(chsmall,150),V.total(ch0,150)),1e-12)

out=dict(passed=all(x['passed']for x in checks),checks=checks,selected_cross_sections=rows,heavy_limits=heavy,runtime_seconds=time.monotonic()-start,model='vacuum exact relativistic two-body point V-A nucleons; F2=GP=0; no finite-T or medium',source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest()for f in(P/'vacuum_cc.py',Path(__file__))})
(P/'results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(passed=out['passed'],checks=len(checks),seconds=out['runtime_seconds'],failed=[x for x in checks if not x['passed']])))
if not out['passed']:raise SystemExit(1)
