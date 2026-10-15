"""Independent arithmetic audit of immutable finite-boundary evidence."""
from pathlib import Path
import json,math,hashlib
import numpy as np
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];OWN=Path(__file__).parent;P=ROOT/'lepton_scattering_box';checks=[]
def ck(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
raw=json.loads((P/'results.json').read_text());base=raw['grids'][-1];s=json.loads((P/'sensitivity_results.json').read_text());hot=[x for x in s['cases']if x.get('label')=='upper_extension'and x['Tr']==3.]
for x in hot:
 g=x['grid'];n=8;E=np.array(g['E']);w=np.array(g['w']);W=np.array(g['W']);A=np.array(g['A']);f=expit((-.7-E)/3);F=A*f[:,None]*(1-f[None,:]);dn=F.sum(axis=0)-F.sum(axis=1);forward=F[:n,n:].sum();reverse=F[n:,:n].sum()
 for key in ['E','w','W']:ck('core '+key+' bitwise upper/order '+str((x['upper'],x['tail_order'])),float(not np.array_equal(np.array(g[key])[:n],base[key])),0)
 ck('core K bitwise '+str((x['upper'],x['tail_order'])),float(not np.array_equal(np.array(g['K'])[:n,:n],base['K'])),0)
 ck('actual missing core number exchange '+str((x['upper'],x['tail_order'])),(dn[:n].sum()-(reverse-forward))/F.sum(),2e-15)
 ck('whole extended box number sum '+str((x['upper'],x['tail_order'])),dn.sum()/F.sum(),2e-15)
 ck('global radiation vs pair heat '+str((x['upper'],x['tail_order'])),(-E@dn/x['observables']['initial_heat']-1),3e-14)
a=next(x for x in hot if x['upper']==36. and x['tail_order']==4)['observables']['final_heat'];b=next(x for x in hot if x['upper']==48. and x['tail_order']==4)['observables']['final_heat'];fine=json.loads((P/'refined_edge_grid_results.json').read_text());h24=next(x['matter_heat']for x in fine['runs']if x['Tr']==3.)
change24=b/h24-1;change36=b/a-1;ck('upper24to48 heat change rounds19.5percent',change24-.195,.0006);ck('upper36to48 heat change rounds1.16percent',change36-.0116,.0001)
r=dict(checks=checks,passed=sum(x['passed']for x in checks),failed=[x for x in checks if not x['passed']],hot_heat24=h24,hot_heat36=a,hot_heat48=b,relative_upper24to48=change24,relative_upper36to48=change36)
(OWN/'boundary_results.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
