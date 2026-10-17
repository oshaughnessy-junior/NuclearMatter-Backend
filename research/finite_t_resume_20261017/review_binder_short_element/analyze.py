"""Saved-only independent stage/rule/raw-moment ledger reconstruction."""
from pathlib import Path
import json,math,hashlib
P=Path(__file__).resolve().parent;A=P.parent/'binder_short_element';PREP=P.parent/'binder_element_review_preparation';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();checks=[];rows=[]
assert sha(PREP/'LOCK.json')=='a647ebe709c66bd0ee92743028f7b8a9efaed4ade18977c811c75e6f1103dee3'
def ck(name,error,gate,category):checks.append(dict(name=name,error=float(error),gate=gate,category=category,passed=bool(abs(error)<=gate)))
def sums(source):
 raw=source['channel_results'];r=[x['net'][0]['value']for x in raw];q=math.fsum(x['net'][1]['value']for x in raw);gR=math.fsum(x[d][0]['value']for x in raw for d in ['forward','backward']);gQ=math.fsum(x[d][1]['value']for x in raw for d in ['forward','backward']);mat=0
 for x in raw:
  v=[y['value']for y in x['net']];mat+=v[4]-v[3]-v[2]if x['channel']=='EC'else v[3]-v[4]-v[2]if x['channel']=='PC'else v[3]+v[2]-v[4]
 return dict(R=r[1]+r[2]-r[0],Q=q,nue=r[0],anti=r[1]+r[2],grossR=gR,grossQ=gQ,material=mat)
for N in [2,4]:
 ctx=json.loads((P/f'trajectory_{N}.json').read_text());assert ctx['complete'];stages=[json.loads((P/n).read_text())for n in ctx['stage_paths']];h=2e-4/N;n=1e-7
 accum={k:[]for k in ['Q','nue','anti','grossR','grossQ','material','entropy','entropygross']}
 for i,z in enumerate(stages):
  v=sums(z['source']);s=z['state'];T=s['temperature_mev'];aff=s['proton']['chemical_potential']+s['electron']['chemical_potential']-s['neutron']['chemical_potential'];entropy=(-v['Q']-aff*v['R'])/(n*T);eg=(v['grossQ']+abs(aff)*v['grossR'])/(n*T)
  ck(f'Heun{N}stage{i} completechannelidentity',0 if z['source']['complete_channels']==['EC','PC','NEUTRON_DECAY']else 1,0,'admission')
  ck(f'Heun{N}stage{i} exactemptyradiation',0 if z['source']['radiation']==dict(mode='empty',temperature=None,mu_nue=None,mu_antinue=None)else 1,0,'admission')
  ck(f'Heun{N}stage{i} root energy achieved',z['root_residual'],z['root_gate'],'root')
  ck(f'Heun{N}stage{i} raw R assembly',(v['R']-z['R'])/v['grossR'],3e-9,'assembly')
  ck(f'Heun{N}stage{i} raw Q assembly',(v['Q']-z['Q'])/v['grossQ'],3e-9,'assembly')
  ck(f'Heun{N}stage{i} raw species energy',(v['material']+v['Q'])/v['grossQ'],3e-6,'source_ledger')
  for key in ['Q','nue','anti','grossR','grossQ','material']:accum[key].append(h*v[key]/2)
  accum['entropy'].append(h*entropy/2);accum['entropygross'].append(h*eg/2)
  resource=json.loads((P/f'RESOURCE_{N}_{i}.json').read_text());lim=resource['usage']['rateLimitsByLimitId']['codex']['primary'];ck(f'Heun{N}stage{i} pregroupquota',0 if resource['admitted']and lim['usedPercent']<97 and lim['resetsAt']==1792652325 else 1,0,'resources')
  ck(f'Heun{N}stage{i} no endpoint source relabel',0 if z['role']==('current corrected state'if i%2==0 else'predictor endpoint, not corrected endpoint')else 1,0,'admission')
 E={k:math.fsum(v)for k,v in accum.items()};initial=ctx['initial'];end=ctx['corrected'][-1]['state'];de=end['energy_density']-initial['energy_density'];le=n*(end['Ye']-initial['Ye'])+E['nue']-E['anti'];ds=end['entropy_density']/n-initial['entropy_density']/n
 ck(f'Heun{N} endpoint independent energy ledger',(de+E['Q'])/E['grossQ'],3e-6,'Heun_energy')
 ck(f'Heun{N} endpoint rawspecies energy ledger',(de-E['material'])/E['grossQ'],3e-6,'Heun_energy')
 ck(f'Heun{N} Le ledger',le/E['grossR'],3e-9,'Le')
 ck(f'Heun{N} materialentropy endpoint-integral',(ds-E['entropy'])/E['entropygross'],1e-4,'entropy')
 for j,key in enumerate(['Q','nue','anti','entropy']):ck(f'Heun{N} storedaccumulator vs raw {key}',(ctx['acc'][j]-E[key])/(E['entropygross']if key=='entropy'else E['grossQ']if key=='Q'else E['grossR']),3e-9,'assembly')
 for i,z in enumerate(ctx['corrected']):ck(f'Heun{N}corrected{i} root energy',z['root_residual'],max(2e-18,1e-8*abs(h*stages[2*i+1]['Q'])), 'root')
 rows.append(dict(method='Heun energy-density/Ye',steps=N,T=end['temperature_mev'],Ye=end['Ye'],escaped_energy=E['Q'],escaped_nue=E['nue'],escaped_anti=E['anti'],energy_defect=de+E['Q'],energy_defect_over_gross= (de+E['Q'])/E['grossQ'],material_entropy_change=ds,material_entropy_integral=E['entropy'],material_entropy_residual_over_gross=(ds-E['entropy'])/E['entropygross'],lepton_residual_over_gross=le/E['grossR'],gross=E))
a,b=rows
for key,gate in [('T',2e-6),('Ye',2e-6)]:
 ck('Heun2vs4 endpoint '+key,a[key]-b[key],gate,'time')
 initial=2 if key=='T'else .3;ck('Heun2vs4 change-normalized '+key,(a[key]-b[key])/abs(b[key]-initial),2e-3,'time')
for key in ['escaped_energy','escaped_nue','escaped_anti']:ck('Heun2vs4 '+key,a[key]/b[key]-1,3e-5,'time')
authorrows=[]
for N in [2,4]:
 x=json.loads((A/f'rung_{N}.json').read_text());h=2e-4/N;vs=[]
 for entry in x['history']:
  z=json.loads((A/f"group_{entry['group']}.json").read_text());vs.append(sums(z['source']));ck(f'AuthorEuler{N}source{entry["stage"]} identity',0 if z['source_identity']==entry['source_identity']and z['state_identity']==entry['state_identity']else 1,0,'author_saved_admission')
 E={key:math.fsum(h*v[key]for v in vs)for key in ['Q','nue','anti','grossR','grossQ','material']};end=x['endpoint'];initial=x['initial'];de=end['energy_density']-initial['energy_density'];res=(de+E['Q'])/E['grossQ']
 ck(f'AuthorEuler{N} energy defect versus independent prelock',res,3e-6,'AuthorEuler_accuracy')
 ck(f'AuthorEuler{N} lepton ledger',(n*(end['Ye']-initial['Ye'])+E['nue']-E['anti'])/E['grossR'],3e-9,'Le')
 authorrows.append(dict(steps=N,T=end['temperature_mev'],Ye=end['Ye'],escaped_energy=E['Q'],energy_defect=de+E['Q'],energy_defect_over_gross=res,author_original_gate=0.02,review_locked_gate=3e-6,accuracy_qualified=False))
f=authorrows[1]
for key,gate in [('T',2e-6),('Ye',2e-6)]:
 ck('AuthorEuler4vsHeun4 '+key,f[key]-b[key],gate,'time')
 initial=2 if key=='T'else .3;ck('AuthorEuler4vsHeun4 change-normalized '+key,(f[key]-b[key])/abs(b[key]-initial),2e-3,'time')
result=dict(kind='Saved-only stage and source-moment reconstruction; all actual scientific calls already complete',checks=checks,passed=sum(x['passed']for x in checks),failed=[x for x in checks if not x['passed']],reference_rows=rows,author_rows=authorrows,classification='Selected Heun self-consistency/endpoint agreement accepted; author Euler energy accuracy UNQUALIFIED under independent3e-6 gate',fresh_source_groups=12,fresh_event_channel_calls=36,extra_scientific_calls_in_analysis=0,order_certificate=False)
(P/'RESULTS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print('PASS',result['passed'],'FAIL',len(result['failed']));print(json.dumps(result['failed'],indent=2));print(json.dumps(rows,indent=2))
