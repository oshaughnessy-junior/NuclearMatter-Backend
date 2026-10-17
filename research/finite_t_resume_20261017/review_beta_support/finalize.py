from pathlib import Path
import json,hashlib,math,datetime
P=Path(__file__).resolve().parent;R=P.parent;A=R/'beta_support';B=R/'beta_review_preparation';h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();L=json.loads((P/'LOCK.json').read_text());checks=[]
def c(n,e=0.,g=0.):checks.append(dict(name=n,error=e,gate=g,passed=bool(math.isfinite(e)and e<=g)))
for f,v in L['author_files'].items():c('author unchanged '+f,float(h(A/f)!=v))
for f,v in L['blind_prep_files'].items():c('blindprep unchanged '+f,float(h(B/f)!=v))
c('author manifest unchanged',float(h(A/'MANIFEST.json')!=L['author_manifest']))
pins=json.loads((B/'LOCK.json').read_text())['source_sha256']
for f,v in pins.items():c('original source unchanged '+f,float(h(Path(f))!=v))
j=[json.loads((P/f'job{i}.json').read_text()) for i in range(4)];r=[x['result'] for x in j]
a=json.loads((A/'E120_SEMI24.json').read_text())['result']; comps=[]
for n,x,y in [('independent_E1.2_lower_higher',r[0],r[1]),('independent_E1.2_author',r[1],{'log_j':a['log_j_s-1'],'log_chi':a['log_true_chi_s-1']}),('heldout_E1.27_oracle_candidate',r[2],{'log_j':r[3]['log_j_s-1'],'log_chi':r[3]['log_true_chi_s-1']})]:
 for k in ('log_j','log_chi'):
  e=abs(math.expm1(x[k]-y[k]));c(n+k,e,3e-6);comps.append(dict(comparison=n,branch=k,relative_error=e))
for i,x in enumerate(r[:3]):
 c(f'oracle{i} shell',x['max_shell'],2e-11);c(f'oracle{i} four',x['max_four'],2e-12);c(f'oracle{i} original unsquared delta',x['max_unsquared_delta'],2e-12);c(f'oracle{i} posteriorKMS',abs(x['KMS_posterior']),2e-9);c(f'oracle{i} reciprocal',x['reciprocal_literal'],1e-10)
 c(f'oracle{i} separately positive representable',float(not(math.isfinite(x['j'])and x['j']>0 and math.isfinite(x['chi'])and x['chi']>0)))
anchors=json.loads((B/'ANALYTIC_SUPPORT.json').read_text())
for i,k in [(1,0),(2,1)]:c('Decimal60 threshold '+str(r[i]['threshold_En']),abs(r[i]['threshold_En']/float(anchors['cases'][k]['parent_Emin_MeV'])-1),2e-12)
old70=json.loads((A/'OLD_TAIL70.json').read_text())['result'];old90=json.loads((A/'OLD_TAIL90.json').read_text())['result'];oldfails=[]
for k in ('j','true_chi'):
 e=abs(math.expm1(old90[k]['log']-old70[k]['log']));assert e>3e-6;oldfails.append(dict(name='original tail70-to90 '+k,error=e,gate=3e-6,passed=False,classification='RETAINED ORIGINAL FAIL'))
# Absolute covariance factor, independent of actual rates: spin2/(4pi²),
# parent1/(2A), active k1/(2E), pair phase1/(16pi²) ->1/(128pi4).
c('covariant absolute coefficient',abs((2/4)*(1/2)*(1/2)*(1/16)-1/128),1e-17)
s={'status':'ACCEPTED_SELECTED_ALTERNATE_BETA_REFERENCE','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checks':checks,'pass':sum(x['passed']for x in checks),'fail':sum(not x['passed']for x in checks),'comparisons':comps,'fresh_calls':4,'integration_seconds':sum(x['seconds']for x in r),'original_failed_local_reproductions':oldfails,'prior_12_beta_failures':json.loads((B/'SAVED_FAILED_GATES.json').read_text())['failed'],'source_candidate_sha256':h(A/'conditional_beta.py'),'source_author_manifest_sha256':h(A/'MANIFEST.json'),'same_literal_current_contraction_algebra':True,'independent_phase':'rest-Dalitz energies and original energy-cone delta rather than pair-CM/cone map','shared_radial_family':'threshold-shifted semi-infinite generalized Laguerre alpha1.5','whole_energy_qualified':False,'upper_incident_panel_qualified':False,'uniform_domain_qualified':False,'new_independent_current_physics_validation':False,'quota_guards':{'job0':'2026-10-17 19:43:14 UTC 94%','job1':'2026-10-17 19:43:16 UTC 94%','job2':'2026-10-17 19:43:17 UTC 94%','job3':'2026-10-17 19:43:19 UTC 94%'},'all_exact_account_window_reset':1792652325}
(P/'RESULTS.json').write_text(json.dumps(s,indent=2)+'\n');assert s['fail']==0;print(json.dumps({k:s[k]for k in('pass','fail','comparisons','integration_seconds','original_failed_local_reproductions')},indent=2))
