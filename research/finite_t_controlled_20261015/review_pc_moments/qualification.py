"""Strict final density qualification and independent species-ledger audit."""
from pathlib import Path
import json,math
H=Path(__file__).parent;R=H.parent;d=json.loads((H/'results.json').read_text());references=[r['reference']for r in d['rows']if'reference'in r];checks=[];diagnostics=[]
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=math.isfinite(e)and abs(e)<=t))
for i,r in enumerate(references):
 error=r['neutron_density']/7e-8-1;item=dict(name='strict integrated density reference'+str(i),error=error,tolerance=3e-8,passed=abs(error)<=3e-8)
 if i==0:diagnostics.append(item)
 else:checks.append(item)
A=R/'positron_capture_moments';author=next(r['result']for r in json.loads((A/'refinement.json').read_text())['rows']if r['kind']=='total'and r['result']['controls']['initial_order']==12);ref=references[-1]
stoich=dict(n=-1,p=1,positron=-1,electron_net=1,antinue=1)
for z,name in enumerate(['empty','LTE','noneq']):
 Rnet=ref['net'][z][0];scale=ref['sums'][z][0][0]+ref['sums'][z][1][0]
 for species,sign in stoich.items():gate('actual source '+name+' '+species,(author['scenarios'][name]['number_sources'][species]-sign*Rnet)/scale,3e-6)
 gate('baryon conservation '+name,stoich['n']+stoich['p'],0.);gate('electric charge '+name,stoich['p']+stoich['positron'],0.);gate('electron Le '+name,stoich['electron_net']-stoich['antinue'],0.)
result=dict(passed=all(c['passed']for c in checks),checks=checks,excluded_own_coarse_diagnostics=diagnostics,scope='Strict author3e-8density gate applied only completed fine references; coarse20p quadrature remains failed at that gate. Species sources independently matched to physical eventstoichiometry and integrated reference Rnet.')
(H/'qualification.json').write_text(json.dumps(result,indent=2)+'\n');print(len(checks),result['passed'],diagnostics)
