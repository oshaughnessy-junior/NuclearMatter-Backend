from pathlib import Path
import json,math,hashlib,datetime,copy
P=Path(__file__).resolve().parent;R=P.parent;A=R/'beta_band_refinement';B=R/'beta_band_review_preparation';h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();L=json.loads((P/'LOCK.json').read_text());ch=[]
def ck(n,e=0.,g=0.):ch.append(dict(name=n,error=e,gate=g,passed=bool(math.isfinite(e)and e<=g)))
for f,v in L['band_author_files'].items():ck('author bytehash '+f,float(h(A/f)!=v))
for f,v in L['prep_files'].items():ck('blindprep bytehash '+f,float(h(B/f)!=v))
ck('blindprepcheckpoint',float(h(B/'CHECKPOINT.json')!=L['prep_sha256']));ck('authormanifest',float(h(A/'MANIFEST.json')!=L['band_author_manifest']))
AL=json.loads((A/'LOCK.json').read_text());AN=json.loads((A/'ANALYSIS.json').read_text());ar=[]
assert AL['source_radiation']==[{'mode':'empty'},{'mode':'FD','T':2.,'mu_antinue':0.}]

def validate(row,job):
 if row.get('complete') is not True or row.get('job')!=job or row.get('lock_sha256')!=h(A/'LOCK.json') or row.get('candidate_sha256')!='148c99c37dc0550ca983b53d86123e25125469b15fd2fde9edc16c169c2ac191':raise ValueError('incomplete/stale/mismatched actual row')
 z=row['result']
 if z.get('bath')!=AL['bath'] or z.get('E_MeV')!=job['E_MeV'] or z.get('rule')!=job['kernel_rule']:raise ValueError('physicalstate/rule/energy mismatch')
 if not all(isinstance(z.get(k),float)and math.isfinite(z[k])for k in('log_j_s-1','log_true_chi_s-1')):raise ValueError('positive finite logs required')
 return z
for job in AL['jobs']:
 f=A/'raw'/f"{job['id']:03d}.json";assert h(f)==L['band_author_files'][str(f.relative_to(A))]
 x=json.loads(f.read_text());z=validate(x,job);ar.append((job,z));ck('author actual row '+str(job['id']))
base=[q for q in ar if q[0]['type']=='baseline'];assert len(base)==64
for order in (12,20,32):
 rows=[q for q in base if q[0]['incident_order']==order];assert len(rows)==order
 for job,z in rows:
  k=job['incident_index'];assert job['E_MeV']==AL['incident_rules'][str(order)]['nodes_MeV'][k] and job['weight_MeV']==AL['incident_rules'][str(order)]['weights_MeV'][k]>0
 ck('complete positive author rule '+str(order))
# Rejection is review-side data admission, not a claimed public API.
original=json.loads((A/'raw/000.json').read_text());jj=AL['jobs'][0]
for name,change in [('missing_complete',lambda x:x.pop('complete')),('source_changed',lambda x:x.update(candidate_sha256='0'*64)),('energy_changed',lambda x:x['result'].update(E_MeV=1.2)),('bath_changed',lambda x:x['result']['bath'].update(T=3.)),('finite_falsezero',lambda x:x['result'].update(**{'log_j_s-1':None})),('weight_changed',lambda x:x['job'].update(weight_MeV=0.))]:
 x=copy.deepcopy(original);change(x)
 try:validate(x,jj)
 except(ValueError,KeyError,TypeError):ck('review admission rejects '+name)
 else:ck('review admission rejects '+name,1.)
rr=[json.loads((P/f'job{i}.json').read_text())for i in range(28)]
for i,x in enumerate(rr):
 assert x['complete'] is True and x['job']==L['jobs'][i];z=x['result']
 ck('oracle shell '+str(i),z['max_shell'],2e-11);ck('oracle four '+str(i),z['max_four'],2e-12);ck('oracle delta '+str(i),z['max_unsquared_delta'],2e-12)
 ck('oracle positivebranches '+str(i),float(not(z['j']>0 and z['chi']>0)))

def aggregate(rows):
 # Distinct from author logsumexp assembler: independently scale each list,
 # sum positive terms with Python math.fsum, and restore the natural log.
 def logsummed(v):
  a=max(v);return a+math.log(math.fsum(math.exp(x-a)for x in v))
 out={}
 for mode in ('empty','FD_mu0'):
  lists={d:{m:[]for m in('number','energy')}for d in('forward','backward')}
  for E,w,lj,lc in rows:
   assert math.isfinite(lj)and math.isfinite(lc)and w>0
   common=math.log(w)+2*math.log(E)-math.log(2*math.pi**2*197.3269804**3)
   ff=1/(1+math.exp(E/2)) if mode=='FD_mu0' else 0.
   for d,l in [('forward',lj+math.log1p(-ff)),('backward',None if ff==0 else lc+math.log(ff))]:
    if l is not None:
     lists[d]['number'].append(common+l);lists[d]['energy'].append(common+l+math.log(E))
  out[mode]={d:{m:({'log':logsummed(v),'value':math.exp(logsummed(v))}if v else{'log':None,'value':0.})for m,v in vals.items()}for d,vals in lists.items()}
 return out
ind=aggregate([(x['job']['E'],x['job']['weight_MeV'],x['result']['log_j'],x['result']['log_chi'])for x in rr[:24]])
auth={str(n):aggregate([(j['E_MeV'],j['weight_MeV'],z['log_j_s-1'],z['log_true_chi_s-1'])for j,z in base if j['incident_order']==n])for n in(12,20,32)}
comps=[];retained=[]
for mode in('empty','FD_mu0'):
 for d in('forward','backward'):
  for m in('number','energy'):
   target=auth['32'][mode][d][m]['log'];a=ind[mode][d][m]['log']
   if target is None:
    ck('exact structuralempty '+mode+d+m,float(a is not None or ind[mode][d][m]['value']!=0));continue
   e=abs(math.expm1(a-target));ck('independent24vs32 '+mode+d+m,e,3e-6);comps.append(dict(mode=mode,direction=d,moment=m,relative_error=e))
   ck('author32 savedarithmetic '+mode+d+m,abs(math.expm1(target-AN['positive_branch_results']['32'][mode][d][m]['log'])),2e-12)
   ck('author20to32 '+mode+d+m,abs(math.expm1(auth['20'][mode][d][m]['log']-target)),3e-6)
   ee=abs(math.expm1(auth['20'][mode][d][m]['log']-auth['12'][mode][d][m]['log']));assert ee>3e-6;retained.append(dict(name=mode+d+m,error=ee,gate=3e-6,passed=False))
edge=[]
for i in range(24,28):
 baseline=rr[L['jobs'][i]['node_index']]['result'];z=rr[i]['result']
 for branch in('log_j','log_chi'):
  e=abs(math.expm1(z[branch]-baseline[branch]));ck('edge '+str(i)+branch,e,3e-6);edge.append(dict(job=i,axisgroup=L['jobs'][i]['role'],branch=branch,relative_error=e))
assert len(retained)==6 and len(AN['failed_checks_preserved'])==6
s={'status':'ACCEPTED_SELECTED_FINITE_INTERVAL_HIGHER_RULES','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checks':ch,'pass':sum(x['passed']for x in ch),'new_fail':sum(not x['passed']for x in ch),'independent_incident24':ind,'independently_recomputed_author_rules':auth,'independent24_vs_author32':comps,'edge_checks':edge,'six_author_coarse_failures_retained':retained,'all_prior_12_beta_failures_remain_failed':True,'fresh_calls':28,'integration_seconds':sum(x['result']['seconds']for x in rr),'same_current_contraction_algebra':True,'shared_parent_radial_family':True,'whole_energy_qualified':False,'uniform_inner_domain_qualified':False,'wrapper_initial_failure':'range4 blocked index4 before physicalprocess; exact initialwrapper preserved; changed only admission to prelockedrange28;28actualcalls total','quota_guards':'eachjob fresh exact identity;94%19:56:36..43,95%19:57:01..35;reset1792652325'}
(P/'RESULTS.json').write_text(json.dumps(s,indent=2)+'\n');assert s['new_fail']==0
print(json.dumps({'pass':s['pass'],'new_fail':s['new_fail'],'maxindependentband':max(v['relative_error']for v in comps),'maxedge':max(v['relative_error']for v in edge),'integration_seconds':s['integration_seconds'],'independent24':ind},indent=2))
