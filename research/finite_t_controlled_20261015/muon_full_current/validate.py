"""Bounded full-current forward/component/source-algebra qualification."""
from pathlib import Path
import math,json,time,hashlib
import numpy as np
import current as C
import source_polynomial as S
P=Path(__file__).resolve().parent;start=time.monotonic();checks=[];components=[];cross_sections=[]
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and abs(error)<=tol)))
def rel(a,b):return (a-b)/b
full=C.FormFactors()
for anti in(False,True):
 for ml,label in[(C.V.ME,'e'),(C.V.MMU,'mu')]:
  ch=C.V.Channel(C.V.MP if anti else C.V.MN,C.V.MN if anti else C.V.MP,ml,anti);M=(ch.initial_mass+ch.final_mass)/2
  for E in([10.,50.,150.]if label=='e'else[ch.threshold+1.,150.,300.]):
   for cosine in(-.9,.2,.97):
    p=C.V.momenta(ch,E,cosine);q=p['pb']-p['pa'];t=C.V.dot(q,q);ff=full.values(t,M)
    parts=C.trace(ch,p,ff,decompose=True);source=C.guo_invariants(ch,p,ff);amp=C.trace(ch,p,ff)
    for key,val in source.items():
     # Relative check each nonzero interference; preserve raw small terms.
     gate(f'Guo {key} {anti}{label}E{E}c{cosine}',rel(parts[key],val),3e-7)
    gate(f'component sum {anti}{label}E{E}c{cosine}',rel(sum(parts.values()),amp),2e-12)
    gate(f'Strumia complete polynomial {anti}{label}E{E}c{cosine}',rel(amp,S.amplitude(ch,p,ff)),3e-12)
    gate(f'VP forbidden {anti}{label}E{E}c{cosine}',parts['VP']/amp,2e-13)
    gate(f'FP forbidden {anti}{label}E{E}c{cosine}',parts['FP']/amp,2e-13)
    gate(f'positive full current {anti}{label}E{E}c{cosine}',min(amp,0.),0.)
    point=C.trace(ch,p,(ch.gv,ch.ga,0.,0.))
    gate(f'zero corrections frozenpoint {anti}{label}E{E}c{cosine}',rel(point,C.V.trace_amplitude(ch,p)),3e-13)
    vs=C.vertices(q,M,ff);fdiv=sum(C.V.METRIC[u]*q[u]*vs['F'][u]for u in range(4))
    gate(f'antisymmetric F transverse {anti}{label}E{E}c{cosine}',np.max(abs(fdiv))/max(np.max(abs(q)),1.),2e-13)
    components.append(dict(antineutrino=anti,lepton=label,E=E,cosine=cosine,t=t,form_factors=ff,trace_parts=parts,guo_parts=source,full_amplitude=amp,point_amplitude=point,fractional_full_vs_point=rel(amp,point),Strumia_amplitude=S.amplitude(ch,p,ff)))
  models={'point':C.FormFactors(weak_magnetism=False,pseudoscalar=False,q2_dependence=False),
          'WM':C.FormFactors(pseudoscalar=False,q2_dependence=False),
          'PS':C.FormFactors(weak_magnetism=False,q2_dependence=False),
          'WM_PS':C.FormFactors(q2_dependence=False),'full_FF':full}
  selectedE=150. if label=='mu' else 20.
  totals={name:C.total(ch,selectedE,model,order=32)for name,model in models.items()}
  refined=C.total(ch,selectedE,full,order=64);source=C.total(ch,selectedE,full,order=64,method='source')
  gate(f'full angular refinement {anti}{label}',rel(totals['full_FF'],refined),3e-10)
  gate(f'full source integration {anti}{label}',rel(source,refined),3e-10)
  gate(f'point total frozen regression {anti}{label}',rel(totals['point'],C.V.total(ch,selectedE)),3e-13)
  cross_sections.append(dict(antineutrino=anti,lepton=label,E=selectedE,totals_cm2=totals,full64_cm2=refined,fractional_vs_point={name:rel(v,totals['point'])for name,v in totals.items()}))

# Massless charged-lepton q-current divergence vanishes: PS drops out.
for anti in(False,True):
 ch=C.V.Channel(939.,939.,0.,anti);p=C.V.momenta(ch,150.,-.3);q=p['pb']-p['pa'];ff=full.values(C.V.dot(q,q),939.)
 terms=C.trace(ch,p,ff,decompose=True);amp=C.trace(ch,p,ff)
 for key in('AP','PP'):gate(f'massless PS zero {anti}{key}',terms[key]/amp,3e-13)

frozen=C.ROOT/'muon_vacuum/vacuum_cc.py'
gate('frozen point source unchanged',0. if hashlib.sha256(frozen.read_bytes()).hexdigest()=='7f79ba2f60f90b53cd23c7cc19e9191837b99442a5fb95f298d163d20d372dd9' else 1.,0.)
for name,fn in [('unknownmethod',lambda:C.differential(C.V.Channel(C.V.MN,C.V.MP,C.V.MMU),150.,0.,method='bad')),
               ('mixedcoupling',lambda:C.differential(C.V.Channel(C.V.MN,C.V.MP,C.V.MMU,ga=1.),150.,0.)),
               ('pionpole',lambda:full.values(full.pion_mass**2,939.))]:
 try:fn()
 except ValueError:rejected=True
 else:rejected=False
 gate('reject '+name,0. if rejected else 1.,0.)
out=dict(passed=all(x['passed']for x in checks),checks=checks,components=components,cross_sections=cross_sections,runtime_seconds=time.monotonic()-start,model='Guo2020 vacuum source form factors; no medium/radiative/secondclass terms',source_sha256={str(p.relative_to(C.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in[Path(__file__),P/'current.py',P/'source_polynomial.py',frozen]})
(P/'results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(passed=out['passed'],checks=len(checks),seconds=out['runtime_seconds'],failed=[x for x in checks if not x['passed']])))
if not out['passed']:raise SystemExit(1)
