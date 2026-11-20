from reference import *
checks=[]
def ck(n,e,g):checks.append(dict(name=n,error=float(e),gate=g,passed=bool(math.isfinite(e)and abs(e)<=g)))
a=json.loads((P/'coarse_results.json').read_text());b=json.loads((P/'fine_results.json').read_text());initial=json.loads((P/'initial_fine_results.json').read_text())
for name in b['results']:
 for branch in ['F','B']:
  for i,obs in enumerate(['number','energy']):
   x,y=a['results'][name][branch][i],b['results'][name][branch][i];ck('fresh joint incident/material refine '+name+' '+branch+' '+obs,x/y-1 if y else x,3e-6 if y else 0.)
 # Removal of an above-endpoint one-ulp redundant rapidity panel must not shift selected totals.
 for branch in ['F','B']:
  for i,obs in enumerate(['number','energy']):
   x,y=initial['results'][name][branch][i],b['results'][name][branch][i];ck('own redundant switch repair selected moments '+name+' '+branch+' '+obs,x/y-1 if y else x,1e-13 if y else 0.)
E=b['results']['empty'];fractions=[E['above_rest_endpoint'][i]/E['F'][i]for i in [0,1]]
ck('actual above-rest number positive',int(E['above_rest_endpoint'][0]<=0),0);ck('actual above-rest energy positive',int(E['above_rest_endpoint'][1]<=0),0)
ck('selected omitted1p4to1p6 number tiny',E['beyond_1p4'][0]/E['F'][0],3e-6);ck('selected omitted1p4to1p6 energy tiny',E['beyond_1p4'][1]/E['F'][1],3e-6)
# Initial entropy reporter bug never changed any fresh phase or assembled F/B moment.
c0=json.loads((P/'initial_coarse_results.json').read_text());c1=json.loads((P/'pre_switch_coarse_results.json').read_text())
for x,y in zip(c0['rows'],c1['rows']):
 x=dict(x);y=dict(y);xx=dict(x['phase']);yy=dict(y['phase']);xx.pop('seconds');yy.pop('seconds');x['phase']=xx;y['phase']=yy
 ck('stale-mu reporter repair leaves raw phase bitwise '+str(x['Enu']),int(x!=y),0)
# Literal n,p,e_net,anti stoichiometry applied to actual independently integrated net.
for name,z in b['results'].items():
 R=z['net'][0];stoich=np.array([-1,1,1,1])*R;gross=z['F'][0]+z['B'][0];ck('integrated beta baryon '+name,(stoich[0]+stoich[1])/gross,1e-14);ck('integrated beta charge '+name,(stoich[1]-stoich[2])/gross,1e-14);ck('integrated beta totalLe '+name,(stoich[2]-stoich[3])/gross,1e-14)
before=json.loads((P/'before.json').read_text());ck('all frozen author/upstream bytes unchanged',int(any(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()!=h for k,h in before.items())),0)
(P/'comparison.json').write_text(json.dumps(dict(checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],above_rest_fractions=fractions,fine_empty=E,source_count=len(before)),indent=2)+'\n');print(len(checks),[c for c in checks if not c['passed']],fractions,len(before))
