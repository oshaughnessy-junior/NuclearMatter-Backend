"""Read-only review: direct FD/DOS sums, absolute residual gates, no physics."""
from pathlib import Path
import json,hashlib,math,copy
import numpy as np
P=Path(__file__).resolve().parent;B=P.parent;A=B/'spectral_band_total_closure';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();H=json.loads((P/'HANDOFF_PIN.json').read_text());M=H['author_files'];checks=[];accuracy=[];blocked=[]
def ck(n,e=0.,g=0.):checks.append(dict(name=n,error=float(e),gate=g,passed=bool(math.isfinite(e)and abs(e)<=g)))
def gate(n,e):accuracy.append(dict(name=n,error=float(e),gate=3e-6,passed=bool(math.isfinite(e)and abs(e)<=3e-6)))
def read(n):
 raw=(A/n).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=M[n]:raise ValueError('Changed raw hash BEFOREparse')
 return json.loads(raw)
for n,h in M.items():assert sha(A/n)==h,n
L=read('LOCK.json');SOURCE='Guo2020-v2-full-free-VA-F2-GP-MV840-MA1000-mpi139.57'
for n,h in json.loads((P/'LOCK.json').read_text())['sourcepins'].items():assert sha(B/n)==h,n
ck('253frozenfiles16sourcepins unchanged')
def positive(z):
 if z['support']!='positive_support'or type(z['log'])not in(int,float)or not math.isfinite(z['log']):raise ValueError('Falsezero/unresolved')
 expected=math.exp(z['log'])
 if type(z['value'])is bool or z['value']!=expected or z['reason']!='represented':raise ValueError('Representation')
def panel(ch,n,p):
 lo,hi=L['panels_MeV'][ch][p:p+2];x,w=np.polynomial.legendre.leggauss(n);Es=[float(v)for v in lo+(x+1)*(hi-lo)/2];ws=[float(v)for v in w*(hi-lo)/2];rows=[];missing=[];jobs=[]
 for first in range(0,n,4):
  fn=f'raw/{ch}_{n}_p{p}_{first:02d}.json'
  if fn not in M:missing+=list(range(first,min(first+4,n)));continue
  d=read(fn);jobs.append(dict(path=fn,sha256=M[fn],complete=d['complete']))
  assert d['lock_sha256']==M['LOCK.json']and d['source_pin_sha256']==M['SOURCE_PIN.json']and d['candidate_manifest_sha256']==L['candidate_manifest_sha256']and d['worker_sha256']==M['node_worker.py']and d['common_harness_sha256']==M['common.py']
  assert d['bath']==L['bath']and d['controls']==L['controls_fine']and d['channel']==ch and d['order']==n and d['panel_index']==p and d['all_band_edges_MeV']==L['panels_MeV'][ch]and d['whole_moment_qualified']is False
  assert d['panel_rule']==dict(nodes=Es,weights=ws,lower=lo,upper=hi)and d['first_index']==first and d['last_index_exclusive']==min(first+4,n)
  for r in d['rows']:
   i=r['index'];z=r['result'];assert type(i)is int and first<=i<min(first+4,n)and r['energy_MeV']==Es[i]and r['weight_MeV']==ws[i]
   assert z['channel']==ch and z['bath']==L['bath']and z['controls']==L['controls_fine']and z['energy']==Es[i]and z['complete']is True and z['current_source']==SOURCE and z['units']=='s^-1 per occupation'and z['differential_coordinates']is None
   positive(z['j']);positive(z['true_chi']);rows.append(r)
  got={r['index']for r in d['rows']};missing +=[i for i in range(first,min(first+4,n))if i not in got]
  if d['complete']is not True:assert missing and d['error']['type']=='ArithmeticError'and d['error']['message']=='Final positive-energy baryon shell unresolved'
 assert len({r['index']for r in rows})==len(rows)
 return rows,sorted(missing),jobs

def aggregate(rows,ch,rad):
 if rad not in L['radiation']:raise ValueError('Wrong radiation')
 if not rows:raise ValueError('No zero-filled source from missing band')
 E=np.array([r['energy_MeV']for r in rows]);w=np.array([r['weight_MeV']for r in rows]);j=np.array([r['result']['j']['value']for r in rows]);x=np.array([r['result']['true_chi']['value']for r in rows]);m=w*E**2/(2*math.pi**2*197.3269804**3)
 if rad['mode']=='empty':f=E*0;force=None
 else:
  mu=rad['mu_nue']if ch=='EC'else rad['mu_antinue'];f=1/(1+np.exp((E-mu)/rad['T']));force=np.array([r['result']['j']['log']-r['result']['true_chi']['log']for r in rows])+(E-mu)/rad['T']
 fwd=j*(1-f);back=x*f;F=[float(np.sum(m*E**k*fwd))for k in[0,1]];G=[float(np.sum(m*E**k*back))for k in[0,1]];N=[float(np.sum(m*E**k*(fwd-back)))for k in[0,1]];ent=None if force is None else float(np.sum(m*(fwd-back)*force));return dict(F=F,B=G,N=N,entropy=ent)
def compare_saved(z,a,name):
 assert a['units']==['fm^-3 s^-1','MeV fm^-3 s^-1']and a['whole_moment_qualified']is False
 for br,key in[('F','forward'),('B','backward')]:
  for k in[0,1]:
   v=z[br][k];av=a[key][k]['value'];ck(name+br+str(k),0 if v==av==0 else av/v-1,3e-12)
 for k in[0,1]:ck(name+'net'+str(k),(a['net'][k]['value']-z['N'][k])/(z['F'][k]+z['B'][k]),3e-12)
 if z['entropy']is not None:ck(name+'entropy',(a['entropy']['value']-z['entropy'])/(z['F'][0]+z['B'][0]),3e-12)
def compare_accuracy(z,y,label):
 for br in['F','B']:
  for k in[0,1]:
   if z[br][k]==y[br][k]==0:continue
   gate(label+br+str(k),z[br][k]/y[br][k]-1)
allrows={};totals={};paneltot={};eventtot={};sums={};eventdiff={};PCmissing=[]
for ch in L['panels_MeV']:
 saved=read(f'{ch}_BAND_RESULTS.json');assert len(saved)==6
 for n in[12,20]:
  allband=[]
  for p in range(5):
   rows,missing,jobs=panel(ch,n,p);allrows[ch,n,p]=rows;allband+=rows
   for rad in L['radiation']:
    sr=next(v for v in saved if v['order']==n and v['radiation']==rad);sp=sr['panels'][p]
    assert sr['bath']==L['bath']and sr['controls']==L['controls_fine']and sr['band_edges_MeV']==L['panels_MeV'][ch]and sr['whole_moment_qualified']is False and sr['source_pin_sha256']==M['SOURCE_PIN.json']and sr['lock_sha256']==M['LOCK.json']
    assert sp['missing_indices']==missing and sp['complete']==(not missing)and sp['actual_job_pins']==jobs
    if missing:assert sp['moments']is None
    else:
     z=aggregate(rows,ch,rad);paneltot[ch,n,p,rad['name']]=z;compare_saved(z,sp['moments'],f'{ch}{n}panel{p}{rad["name"]}')
   if missing:PCmissing.append(dict(channel=ch,order=n,panel=p,missing_indices=missing,actual_rows=len(rows)))
  for rad in L['radiation']:
   sr=next(v for v in saved if v['order']==n and v['radiation']==rad)
   if any(len(allrows[ch,n,p])!=n for p in range(5)):
    assert sr['complete_declared_band']is False and sr['band_moments']is None;ck(f'{ch}{n}{rad["name"]}incompletenotzero');continue
   assert sr['complete_declared_band']is True;z=aggregate(allband,ch,rad);totals[ch,n,rad['name']]=z;compare_saved(z,sr['band_moments'],f'{ch}{n}BAND{rad["name"]}')
 for rad in L['radiation']:
  e=read(f'events/{ch}_{rad["name"]}.json');r=e['result'];assert e['complete']is True and e['channel']==ch and e['bath']==L['bath']and e['radiation']==rad and e['lock_sha256']==M['LOCK.json']and e['source_pin_sha256']==M['SOURCE_PIN.json']and e['candidate_manifest_sha256']==L['candidate_manifest_sha256']and e['actual_controls']==r['controls']
  assert r['bath']==L['bath']and r['channel']==ch and r['current']==dict(source=SOURCE,neutron_mass=939.56542052,proton_mass=938.27208816,electron_mass=.51099895069,weak_mass=938.91875434)and r['radiation']==dict(mode=rad['mode'],temperature=rad['T'],mu_nue=rad['mu_nue'],mu_antinue=rad['mu_antinue'])
  assert e['names']==['number','neutrino_energy','charged_lepton_energy','proton_energy','neutron_energy','material_energy_loss']and e['units']=='number fm^-3 s^-1; energy MeV fm^-3 s^-1; entropy kB fm^-3 s^-1'
  ev=dict(F=[v['value']for v in r['forward'][:2]],B=[v['value']for v in r['backward'][:2]]);eventtot[ch,rad['name']]=ev;N=[v['value']for v in r['net']];g=r['forward'][1]['value']+r['backward'][1]['value'];loss=N[4]-N[3]-N[2]if ch=='NEUTRON_DECAY'else N[2]+N[3]-N[4]if ch=='EC'else N[2]+N[4]-N[3];ck(f'{ch}{rad["name"]}eventspeciesE',(loss-N[1])/g,3e-9);ck(f'{ch}{rad["name"]}directloss',(N[5]-N[1])/g,3e-9)
  st=dict(r['particle_stoichiometry']);ck(f'{ch}{rad["name"]}baryonchargeLe',st['neutron']+st['proton']);ck(f'{ch}{rad["name"]}charge',st['proton']-st['electron_net']);ck(f'{ch}{rad["name"]}Le',st['electron_net']+(st.get('nue',0)-st.get('antinue',0)))
  if(ch,20,rad['name'])not in totals:blocked.append(dict(channel=ch,radiation=rad['name'],reason='incomplete band, no total source'));continue
  z=totals[ch,20,rad['name']];compare_accuracy(totals[ch,12,rad['name']],z,f'{ch}BAND12to20{rad["name"]}');compare_accuracy(z,ev,f'{ch}BANDvsEVENT{rad["name"]}');eventdiff[ch,rad['name']]=dict(F=[z['F'][k]/ev['F'][k]-1 for k in[0,1]],B=[0 if ev['B'][k]==0 else z['B'][k]/ev['B'][k]-1 for k in[0,1]])
  for p in range(5):compare_accuracy(paneltot[ch,12,p,rad['name']],paneltot[ch,20,p,rad['name']],f'{ch}panel{p}12to20{rad["name"]}')
  if rad['mode']=='FD':
   eq=L['bath']['mu_p']+L['bath']['mu_e']-L['bath']['mu_n'];eq=eq if ch=='EC'else-eq;mu=rad['mu_nue']if ch=='EC'else rad['mu_antinue'];gross=z['F'][0]+z['B'][0];ck(f'{ch}{rad["name"]}entropyforce',(z['entropy']-(eq-mu)/rad['T']*z['N'][0])/gross,3e-9)
   if rad['name']=='LTE':ck(ch+'LTEgross',z['N'][0]/gross,2e-9)
# Recompute literal spot comparison errors; do not accept a negative signed
# relative difference solely because it is below a positive gate.
controls=read('CONTROL_ASSESSMENT.json');base={};control_results={}
for c in controls:
 d=read(c['path']);assert d['channel']==c['channel']and d['energy_MeV']==c['E']and d['bath']==L['bath']and d['variation']==c['variation']and d['controls']==dict(L['controls_fine'],**c['variation']['changes'])and d['lock_sha256']==M['LOCK.json']and d['source_pin_sha256']==M['SOURCE_PIN.json']
 key=c['channel'],c['E'];name=c['variation']['name'];control_results[key+(name,)]=d
 if not c['complete']:assert not d['complete']and d['error']['message']=='Final positive-energy baryon shell unresolved';blocked.append(dict(channel=c['channel'],control=name,reason=d['error']));continue
 assert d['complete'];result=d['result'];positive(result['j']);positive(result['true_chi'])
 if name=='fine':base[key]=result
for key,d in control_results.items():
 ch,E,name=key
 if not d['complete']or name=='fine':continue
 for br in['j','true_chi']:gate(f'{ch}{E}{name}{br}',math.expm1(d['result'][br]['log']-base[ch,E][br]['log']))
# Frozen author flags must themselves use abs(error), and retain all12failures.
a=read('ANALYSIS.json');assert a['passed']==188 and len(a['failed'])==12 and len(a['blocked'])==6
for c in a['checks']:ck('authorabsflag '+c['name'],0 if c['passed']==bool(math.isfinite(c['error'])and abs(c['error'])<=c['gate'])else 1)
# Actual preregistered replays: expected exception is a reproduced blocker,
# NOT a passing accuracy test; actual positive beta coefficients match bytes.
w=json.loads((P/'WITNESS_RESULTS.json').read_text());assert w['complete']and len(w['rows'])==3
assert not w['rows'][0]['completed']and w['rows'][0]['exception']==dict(type='ArithmeticError',message='Final positive-energy baryon shell unresolved');ck('PC.1 unresolvedreproduced')
for i,name in[(1,'fine'),(2,'isolated_parent_tail90')]:
 z=w['rows'][i];assert z['completed'];ref=control_results['NEUTRON_DECAY',1.2,name]['result'];ck('betawitnessbitwise'+name,0 if z['result']==ref else 1)
replay_errors={br:math.expm1(w['rows'][2]['result'][br]['log']-w['rows'][1]['result'][br]['log'])for br in['j','true_chi']}
ck('noPCnodesfilled',0 if sum(len(r)for k,r in allrows.items()if k[0]=='PC')==0 else 1)
failed=[z for z in accuracy if not z['passed']];ck('preserved12numericFAIL',len(failed)-12);ck('preserved6blocked',len(blocked)-6)
# Synthetic copies are never written back to author/source data.
for rad in[dict(L['radiation'][1],mu_nue=0.),dict(L['radiation'][1],T=2.1)]:
 try:aggregate(allrows['EC',20,0],'EC',rad)
 except ValueError:ck('wrongradiationrejected')
 else:ck('wrongradiationrejected',1)
try:aggregate([],'PC',L['radiation'][0])
except ValueError:ck('zero_fill_missingbandrejected')
else:ck('zero_fill_missingbandrejected',1)
num=lambda z:{k:[float(v)for v in x]if isinstance(x,list)else x for k,x in z.items()};out=dict(passed=sum(z['passed']for z in checks),review_failed=[z for z in checks if not z['passed']],checks=checks,independent_accuracy_checks=accuracy,retained_numeric_failed=failed,blocked=blocked,PC_missing_maps=PCmissing,actual_spectral_nodes={ch:sum(len(r)for k,r in allrows.items()if k[0]==ch)for ch in L['panels_MeV']},witness_tail_relative_errors=replay_errors,band_event_residuals=[dict(channel=k[0],radiation=k[1],residual=v)for k,v in eventdiff.items()],accepted_scope='EC selected band agreement ONLY; PC blocked, beta diagnostic unqualified; no infinite tails',whole_moment_qualified=False)
(P/'RESULTS.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(out['passed'],'reviewPASS',len(out['review_failed']),'reviewFAIL;',len(failed),'numericFAIL',len(blocked),'BLOCKED');print(out['review_failed']);print('BAND_EVENT_MAX',{ch:max(abs(v)for k,x in eventdiff.items()if k[0]==ch for vals in x.values()for v in vals)for ch in['EC','NEUTRON_DECAY']});print('TAIL',replay_errors);print('PANEL_MAX',max(abs(z['error'])for z in accuracy if'panel'in z['name']))
