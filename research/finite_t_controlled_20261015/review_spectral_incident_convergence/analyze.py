"""Independent longdouble ordinary-occupation summation; no physical imports."""
from pathlib import Path
import json,hashlib,math,copy
import numpy as np
P=Path(__file__).resolve().parent;B=P.parent;A=B/'spectral_incident_convergence';sha=lambda x:hashlib.sha256(x).hexdigest();L=json.loads((P/'LOCK.json').read_text());checks=[];coarse=[];records=[];exceptions=[]
def ck(name,error=0.,gate=0.):checks.append(dict(name=name,error=float(error),gate=gate,passed=bool(math.isfinite(error)and abs(error)<=gate)))
def hashparse(p,h):
 raw=p.read_bytes()
 if sha(raw)!=h:raise ValueError('Raw-byte checksum before parsing')
 return json.loads(raw)
M=hashparse(A/'MANIFEST_SUCCESSOR.json',L['study_frozen_hashes']['MANIFEST_SUCCESSOR.json'])['sha256']
def read(rel):return hashparse(A/rel,M[rel])
for n,h in M.items():assert sha((A/n).read_bytes())==h,n
old=read('ORIGINAL_SOURCE_CHECKPOINT_MANIFEST.json');assert len(old)==122
for n,h in old.items():assert sha((A/n).read_bytes())==h,n
ck('122originalcheckpointpathsunchanged')
S=read('LOCK.json');adm=read('SUCCESSOR_SOURCE_ADMISSION.json');assert adm['original_candidate_manifest']=='18da0dfe7f52c8fea7c59d9fe7e1687b5893738c36cf836091587cff02406ad0'and adm['repaired_candidate_manifest']=='dcfbafe85645caa4ac6f2ea1f4cd3a7cb55775d030eddc0c9c5a8c9211a8b216'
for original,r in adm['original_execution']['source_snapshot_mapping'].items():assert sha((B/r['verified_snapshot']).read_bytes())==r['sha256'],original
for n,h in read('SUCCESSOR_SOURCE_PIN.json').items():assert sha((B/n).read_bytes())==h,n
ck('explicitoriginal168versussuccessor84sourceadmission')
SOURCE='Guo2020-v2-full-free-VA-F2-GP-MV840-MA1000-mpi139.57'
def rule(ch,n):
 lo,hi=S['channels'][ch];x,w=np.polynomial.legendre.leggauss(n);return dict(nodes=[float(v)for v in lo+(x+1)*(hi-lo)/2],weights=[float(v)for v in w*(hi-lo)/2],lower=lo,upper=hi)
def positive(v):
 if v['support']!='positive_support'or type(v['log'])not in(int,float)or not math.isfinite(v['log']):raise ValueError('Unresolved/falsezero/nonfinite integrated coefficient')
 val=math.exp(v['log']);reason='underflow'if val==0 else'overflow'if not math.isfinite(val)else'represented'
 expected=val if reason=='represented'else None
 if type(v['value'])is bool or v['value']!=expected or v['reason']!=reason:raise ValueError('Coefficient representation')
def packet(d,ch,n,first,new=False):
 r=L['rules'][ch]if new else rule(ch,n)
 if d['complete']is not True or d['channel']!=ch or d['order']!=n or d['rule']!=r or d['bath']!=S['bath']or d['controls']!=S['controls']:raise ValueError('Incomplete/state/rule/control')
 if new:
  if d['first']!=first or d['stop']!=first+4 or d['lock_sha256']!=sha((P/'LOCK.json').read_bytes())or d['worker_sha256']!=sha((P/'nodes20.py').read_bytes())or d['sourcepins']!=L['sourcepins']:raise ValueError('New source provenance')
 else:
  orig=ch!='NEUTRON_DECAY';pin='SOURCE_PIN.json'if orig else'SUCCESSOR_SOURCE_PIN.json';worker='nodes.py'if orig else'nodes_successor.py';candidate=adm['original_candidate_manifest']if orig else adm['repaired_candidate_manifest']
  if d['first_index']!=first or d['last_index_exclusive']!=min(first+4,n)or d['source_pin_sha256']!=M[pin]or d['worker_sha256']!=M[worker]or d['lock_sha256']!=M['LOCK.json']or d['candidate_manifest_sha256']!=candidate:raise ValueError('Old source/worker/lock')
  if not orig and(d['execution_source_group']!='repaired_guard_successor'or d['source_pin_file']!='SUCCESSOR_SOURCE_PIN.json'or d['successor_admission_sha256']!=M['SUCCESSOR_SOURCE_ADMISSION.json']):raise ValueError('Successor provenance')
 if len(d['rows'])!=min(first+4,n)-first:raise ValueError('Missing rows')
 for i,row in enumerate(d['rows'],first):
  z=row['result']
  if type(row['index'])is not int or row['index']!=i or row['energy']!=r['nodes'][i]or row['weight']!=r['weights'][i]or z['energy']!=row['energy']:raise ValueError('Changed/duplicate index energy weight')
  if z['bath']!=S['bath']or z['controls']!=S['controls']or z['channel']!=ch or z['current_source']!=SOURCE or z['complete']is not True or z['differential_coordinates']is not None or z['units']!='s^-1 per occupation':raise ValueError('Spectral identity')
  positive(z['j']);positive(z['true_chi'])
 return d['rows']
def collect(ch,n):
 out=[]
 for first in range(0,n,4):out+=packet(read(('raw/'if ch!='NEUTRON_DECAY'else'raw_successor/')+f'{ch}_{n}_{first:02d}.json'),ch,n,first)
 assert len(out)==n;return out
# Direct ordinary FD occupations in np.longdouble (52 mantissa bits on this ARM host). This is a
# distinct algorithm from author per-node logsumexp/expm1, at representable data.
D=np.longdouble;HC=D('197.3269804');PI=D(str(math.pi))
def totals(rows,ch,rad):
 if rad not in S['radiation']:raise ValueError('Unregistered radiation potential/closure')
 E=np.array([r['energy']for r in rows],dtype=D);weight=np.array([r['weight']for r in rows],dtype=D);j=np.array([r['result']['j']['value']for r in rows],dtype=D);x=np.array([r['result']['true_chi']['value']for r in rows],dtype=D)
 if rad['mode']=='empty':f=E*0;force=None
 else:
  mu=D(rad['mu_nue']if ch=='EC'else rad['mu_antinue']);f=1/(1+np.exp((E-mu)/D(rad['T'])));force=np.array([D(r['result']['j']['log'])-D(r['result']['true_chi']['log'])for r in rows])+(E-mu)/D(rad['T'])
 a=j*(1-f);b=x*f;measure=weight*E**2/(2*PI**2*HC**3);F=[np.sum(measure*E**k*a,dtype=D)for k in [0,1]];G=[np.sum(measure*E**k*b,dtype=D)for k in [0,1]];N=[np.sum(measure*E**k*(a-b),dtype=D)for k in [0,1]];entropy=None if force is None else np.sum(measure*(a-b)*force,dtype=D)
 return dict(F=F,B=G,N=N,entropy=entropy)
def serial(v):return{key:[float(x)for x in vv]if isinstance(vv,list)else None if vv is None else float(vv)for key,vv in v.items()}
can=read('CANONICAL_FINITE_RANGE_RESULTS.json');author={(r['channel'],r['order'],r['radiation']['name']):r for r in can['profile_aggregates']};assert len(author)==45
allrows={};alltotals={}
for ch in S['channels']:
 for n in S['incident_orders']:
  rows=collect(ch,n);allrows[ch,n]=rows;ck(f'{ch}{n} complete positive nodes')
  for rad in S['radiation']:
   key=(ch,n,rad['name']);z=totals(rows,ch,rad);alltotals[key]=z;a=author[key];assert a['execution_source_group']==('original_18da0'if ch!='NEUTRON_DECAY'else'repaired_guard_successor')and a['execution_candidate_manifest']==(adm['original_candidate_manifest']if ch!='NEUTRON_DECAY'else adm['repaired_candidate_manifest'])and a['source_pin_sha256']==M['SOURCE_PIN.json'if ch!='NEUTRON_DECAY'else'SUCCESSOR_SOURCE_PIN.json']and a['accepted_successor_admission_sha256']==M['SUCCESSOR_SOURCE_ADMISSION.json'];assert all(M[v['path']]==v['sha256']for v in a['actual_job_pins']);ck(f'{key}canonical execution/source/group pins');assert a['bath']==S['bath']and a['controls']==S['controls']and a['radiation']==rad and a['units']==['fm^-3 s^-1','MeV fm^-3 s^-1']and a['whole_moment_qualified']is False
   for branch,field in [('F','forward'),('B','backward')]:
    for k in [0,1]:
     v=z[branch][k];av=a[field][k]['value'];ck(f'{key}{branch}{k} independent ordinarysum',0 if v==0 and av==0 else float(D(av)/v-1),3e-12)
   for k in [0,1]:ck(f'{key}net{k}gross',float((D(a['net'][k]['value'])-z['N'][k])/(z['F'][k]+z['B'][k])),3e-12)
   if rad['mode']=='FD':
    mu=rad['mu_nue']if ch=='EC'else rad['mu_antinue'];eq=S['bath']['mu_p']+S['bath']['mu_e']-S['bath']['mu_n'];eq=eq if ch=='EC'else-eq;aff=D((eq-mu)/S['bath']['T']);gross=z['F'][0]+z['B'][0]
    ck(f'{key}entropyforce',float((z['entropy']-aff*z['N'][0])/gross),3e-9);ck(f'{key}entropy published',float((D(a['entropy']['value'])-z['entropy'])/gross),3e-12);ck(f'{key}entropy nonnegative',min(float(z['entropy']/gross),0.),3e-12)
    if rad['name']=='LTE':ck(f'{key}LTEgross',float(z['N'][0]/gross),2e-9)
    else:ck(f'{key}affinitysign',0 if z['N'][0]>0 and aff>0 else 1)
   # Integer number species reconstruct independent E/Le/charge signs.
   dn,dp,de,lnu=(1,-1,-1,1)if ch=='EC'else(-1,1,1,-1)
   ck(f'{key}baryon charge Le',abs(dn+dp)+abs(dp-de)+abs(de+lnu))
# Reproduce ALL original coarse failures without changing author counts/gates.
for ch in S['channels']:
 for na,nb in zip(S['incident_orders'][:-1],S['incident_orders'][1:]):
  for rad in S['radiation']:
   for br in ['F','B']:
    if br=='B'and rad['mode']=='empty':continue
    for k in [0,1]:
     err=abs(float(alltotals[ch,na,rad['name']][br][k]/alltotals[ch,nb,rad['name']][br][k]-1));v=dict(channel=ch,orders=[na,nb],radiation=rad['name'],branch=br,moment=k,error=err,gate=3e-6,passed=err<=3e-6);coarse.append(v)
# New order20 result is compared ONCE against32; failing gate not reprofiled.
newhashes={};newresults=[]
for ch in S['channels']:
 rows=[]
 for first in range(0,20,4):
  f=P/'raw'/f'{ch}_{first:02d}.json';raw=f.read_bytes();h=sha(raw);newhashes[str(f.relative_to(P))]=h;rows+=packet(hashparse(f,json.loads((P/'NEW_RAW_PIN.json').read_text())['sha256'][str(f.relative_to(P))]),ch,20,first,True)
 for rad in S['radiation']:
  z=totals(rows,ch,rad);ref=alltotals[ch,32,rad['name']];diff=[]
  for br in ['F','B']:
   if br=='B'and rad['mode']=='empty':continue
   for k in [0,1]:
    e=abs(float(z[br][k]/ref[br][k]-1));ck(f'NEW20vs32 {ch}{rad["name"]}{br}{k}',e,3e-6);diff.append(dict(branch=br,moment=k,error=e))
  gross=z['F'][0]+z['B'][0]
  if rad['mode']=='empty':ck(f'NEW20 {ch}empty inverseexactabsent',0 if z['B']==[D(0),D(0)]and z['entropy']is None else 1)
  else:
   eq=S['bath']['mu_p']+S['bath']['mu_e']-S['bath']['mu_n'];eq=eq if ch=='EC'else-eq;mu=rad['mu_nue']if ch=='EC'else rad['mu_antinue'];aff=D((eq-mu)/S['bath']['T']);ck(f'NEW20 {ch}{rad["name"]}entropyforce',float((z['entropy']-aff*z['N'][0])/gross),3e-9);ck(f'NEW20 {ch}{rad["name"]}entropypositive',min(float(z['entropy']/gross),0.),3e-12)
   if rad['name']=='LTE':ck(f'NEW20 {ch}LTEgross',float(z['N'][0]/gross),2e-9)
   else:ck(f'NEW20 {ch}positiveaffinitysign',0 if aff>0 and z['N'][0]>0 else 1)
  # Radiation and matter energy/Le bookkeeping use actual net moments; these
  # algebraic ledgers are not full spectral species-energy validation.
  materialQ=-z['N'][1];radiationQ=z['N'][1];ck(f'NEW20 {ch}{rad["name"]}Eledger',float((materialQ+radiationQ)/(z['F'][1]+z['B'][1])),3e-12)
  lsign=1 if ch=='EC'else-1;materialL=-lsign*z['N'][0];radiationL=lsign*z['N'][0];ck(f'NEW20 {ch}{rad["name"]}Leledger',float((materialL+radiationL)/gross),3e-12)
  newresults.append(dict(channel=ch,order=20,radiation=rad,independent_moments=serial(z),comparison32=diff,scope='FINITErange fixedinner only'))
# Admission adversaries exercise OUR independent validator, and immutable raw
# checksum rejects changed bytes before parsing; no study/source files mutate.
base=read('raw/EC_4_00.json')
mutations={'missingrow':lambda d:d['rows'].pop(),'stale source':lambda d:d.update(source_pin_sha256='0'*64),'changedweight':lambda d:d['rows'][0].update(weight=.1),'changedrule':lambda d:d['rule']['weights'].__setitem__(0,.1),'current':lambda d:d['rows'][0]['result'].update(current_source='x'),'FDstate':lambda d:d['bath'].update(mu_e=.2),'incomplete':lambda d:d.update(complete=False),'logfalsezero':lambda d:d['rows'][0]['result']['j'].update(value=0.,log=None,support='exact_kinematic_zero'),'Boolean':lambda d:d['rows'][0]['result']['true_chi'].update(value=True),'nonfinite':lambda d:d['rows'][0]['result']['j'].update(log=float('nan')),'wrongunits':lambda d:d['rows'][0]['result'].update(units='cm^-1'),'duplicateindex':lambda d:d['rows'][1].update(index=0)}
for name,mut in mutations.items():
 d=copy.deepcopy(base);mut(d)
 try:packet(d,'EC',4,0)
 except ValueError as e:ck('reject '+name);exceptions.append(dict(name=name,error=str(e)))
 else:ck('reject '+name,1)
for rad in [dict(S['radiation'][1],mu_nue=-1.),dict(S['radiation'][1],T=2.1)]:
 try:totals(allrows['EC',4],'EC',rad)
 except ValueError:ck('reject unregistered radiation'+str(rad))
 else:ck('reject unregistered radiation'+str(rad),1)
try:hashparse(A/'raw/EC_4_00.json','0'*64)
except ValueError:ck('reject changedraw BEFOREjsonparse')
else:ck('reject changedraw BEFOREjsonparse',1)
cf=[r for r in coarse if not r['passed']];ck('retained29coarsefailures',len(cf)-29)
counts=dict(old_actual_nodes=sum(len(r)for r in allrows.values()),new_actual_nodes=60,original_ECPC=168,successor_beta=84,coarse_failures=len(cf));out=dict(counts=counts,passed=sum(r['passed']for r in checks),failed=[r for r in checks if not r['passed']],checks=checks,original_convergence_checks=coarse,preserved_original_coarse_failures=cf,new_order20_results=newresults,new_raw_sha256=newhashes,admission_exception_evidence=exceptions,new_physics='60actualsharedinnercoefficients, no extra rule/order/tail',scope='finite incident rules only; no wholeenergy/inner-domain/physicalmodel/CI certificate')
(P/'RESULTS.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(out['passed'],'PASS',len(out['failed']),'newFAIL; retained',len(cf),'oldcoarseFAIL');print(out['failed']);print('NEWMAX',max(v['error']for r in newresults for v in r['comparison32']));print('FINEMAX',{ch:max(r['error']for r in coarse if r['channel']==ch and r['orders']==[24,32])for ch in S['channels']})
