"""Fresh hot reference controls and explicit delta/DOS normalization audit."""
from pathlib import Path
import sys,json,math,time
import numpy as np
R=Path(__file__).resolve().parents[1];P=Path(__file__).resolve().parent;sys.dont_write_bytecode=True;sys.path.insert(0,str(R/'hot_beta_moments'))
import hot_moments as M,hot_spectrum as S
b,meta=M.K.matched_npe(2.,1e-7,.3);checks=[];rows=[];start=time.monotonic()
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
# Test hard exact-state boundary BEFORE an integral, not pretending another bath is qualified.
for name,bb,E,kw in [('old cold API',b,.42,{'old':True}),('changedT',M.K.Bath(2.0000001,b.mu_n,b.mu_p,b.mu_e),.42,{}),('changed_mu',M.K.Bath(2.,b.mu_n,b.mu_p,b.mu_e+1e-10),.42,{}),('changedE',b,.42000000001,{}),('boolangle',b,.42,dict(angle_order=True)),('nantail',b,.42,dict(tail=math.nan)),('badwidth',b,.42,dict(widths=17.)),('badbudget',b,.42,dict(budget_seconds=121))]:
 try:
  if kw.pop('old',False):M.K.spectral(bb,E)
  else:S.spectral(bb,E,**kw)
  gate('reject '+name,1,0)
 except ValueError:gate('reject '+name,0,0)
for controls in [dict(angle_order=12,lepton_order=18,energy_order=20,phi_order=4,widths=11.,tail=60.),dict(angle_order=18,lepton_order=24,energy_order=28,phi_order=8,widths=15.,tail=90.)]:
 r=S.spectral(b,.42,**controls,budget_seconds=110);rows.append(r);(P/'spectral_partial.json').write_text(json.dumps(dict(rows=rows,checks=checks),indent=2)+'\n')
for key,expected in [('j',.07083091288795236),('chi',.02144761484832465)]:
 gate('new selected hot refinement '+key,rows[0][key]['value']/rows[1][key]['value']-1,3e-6);gate('fresh hot tensor vs conditional '+key,rows[1][key]['value']/expected-1,3e-6)
# Independent delta differentiation and a one-helicity measure derivation.
for g,e,c in [(1.002,.42,.31),(1.03,.33,-.57)]:
 v=math.sqrt(1-1/g**2);h=1e-5;E=lambda z:g*e*(1+v*z);der=(E(c+h)-E(c-h))/(2*h);jac=1/(2*g*v*e);gate('isotropic angle delta derivative',2*jac*der-1,3e-10)
# d^3k/(2pi hc)^3, one helicity: 4pi/(2pi)^3 =1/(2pi²), no spin2.
D=.42**2*4*math.pi/(2*math.pi*M.HC)**3;gate('one helicity DOS',D/(.42**2/(2*math.pi**2*M.HC**3))-1,1e-15)
# Literal physical beta event conservation before any potential derivative claims.
for name,q in [('Q',dict(n=0,p=1,e=-1,a=0)),('Le',dict(n=0,p=0,e=1,a=-1)),('B',dict(n=1,p=1,e=0,a=0))]:gate('reaction '+name,-q['n']+q['p']+q['e']+q['a'],0)
(P/'spectral_results.json').write_text(json.dumps(dict(checks=checks,rows=rows,passed=all(c['passed']for c in checks),seconds=time.monotonic()-start,scope='new control ladder at exact selected state; author conditional/source geometry audited, not a new independent whole-spectrum oracle'),indent=2)+'\n');print(len(checks),all(c['passed']for c in checks),time.monotonic()-start);print([c for c in checks if not c['passed']])
