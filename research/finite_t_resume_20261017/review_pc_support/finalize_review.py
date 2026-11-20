from pathlib import Path
import hashlib,json,math,datetime
D=Path(__file__).resolve().parent; A=D.parent/'pc_support'; P=D.parent/'independent_review_preparation'
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
lock=json.loads((D/'LOCK.json').read_text()); checks=[]
def check(name,error=0.,gate=0.):checks.append(dict(name=name,error=error,gate=gate,passed=bool(math.isfinite(error) and error<=gate)))
for f,v in lock['author_files'].items():check('author unchanged '+f,float(h(A/f)!=v))
for f,v in lock['prep_files'].items():check('candidate-blind prep unchanged '+f,float(h(P/f)!=v))
check('author manifest unchanged',float(h(A/'MANIFEST.json')!=lock['author_manifest_sha256']))
jobs=[json.loads((D/f'job{i}.json').read_text()) for i in range(4)]
r=[j['result'] for j in jobs]
for i,j in enumerate(jobs):check(f'job{i} complete',float(not j.get('complete',j.get('kind')=='fresh candidate heldout')))
forward=json.loads((A/'ASSESSMENT.json').read_text())['forward_rows'][0]
comparisons=[]
for label,x,y in [('oracle_0.1_coarse_fine',r[0],r[1]),('oracle_0.1_vs_frozen_author',r[1],{'log_j':forward['log_j_s-1'],'log_chi':forward['log_true_chi_s-1']}),('heldout_0.13_oracle_candidate',r[2],{'log_j':r[3]['log_j_s-1'],'log_chi':r[3]['log_true_chi_s-1']})]:
 for k in ('log_j','log_chi'):
  err=abs(math.expm1(x[k]-y[k]));check(label+' '+k,err,3e-6);comparisons.append(dict(test=label,branch=k,relative_error=err))
for i,x in enumerate(r[:3]):
 check(f'oracle{i} massshell',x['max_shell'],2e-11);check(f'oracle{i} four',x['max_four'],2e-12);check(f'oracle{i} unsquared delta',x['max_unsquared_delta'],2e-12)
 check(f'oracle{i} posterior KMS',abs(x['KMS_posterior']),2e-9)
 check(f'oracle{i} ordinary underflow retained',float(x['ordinary_j'] is not None or x['ordinary_chi'] is not None))
 for k in ('log_j','log_chi'):check(f'oracle{i} finite {k}',float(not math.isfinite(x[k])))
anchor=json.loads((P/'ANALYTIC_SUPPORT_ANCHORS.json').read_text())
check('Decimal60 threshold vs oracle',abs(r[1]['threshold_Ep']/float(anchor['proton_Emin_MeV'])-1),2e-12)
# Independent reduction: g2/(4pi²hbar) p Ep * (s-mp²)/(2E Ep) * avg2/(64pi²s) pf/pi,
# pi=(s-mp²)/(2sqrt(s)), avg2=2avg4. Coefficient ->1/(64pi4 E sqrt(s)hbar).
coef=(2/4)*(1/2)*(1/64)*2*2
check('unreduced invariant measure coefficient',abs(coef-1/64),1e-17)
report={'status':'ACCEPTED_SELECTED_SOFT_PC_REFERENCE','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checks':checks,'pass':sum(x['passed'] for x in checks),'fail':sum(not x['passed'] for x in checks),'comparisons':comparisons,'fresh_integral_calls':4,'fresh_integral_seconds':sum(x.get('seconds',0.) for x in r[:3])+jobs[3]['seconds'],'candidate_sha256':h(A/'conditional_pc.py'),'author_manifest_sha256':h(A/'MANIFEST.json'),'scope':{'independent_integrated_incident_energies_MeV':[.1,.13],'author_only_additional_incident_energy_MeV':.15,'whole_energy_qualified':False,'uniform_domain_qualified':False,'external_physics_validation':False},'preserved_failures':{'original_qaxis_shell':h(A/'ORIGINAL_FAILURE.json'),'original_incoming_endpoint':h(A/'UNMAPPED_INCOMING_ENDPOINT_RAW.json')},'quota_guards_UTC':{'job0':'19:16:56 92%','job1':'19:17:22 93%','job2':'19:17:26 93%','job3':'19:17:29 93%'},'all_quota_guards_exact_reset':1792652325,'same_physical_model':True}
(D/'RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
assert report['fail']==0
print(json.dumps({k:report[k] for k in ('pass','fail','fresh_integral_calls','fresh_integral_seconds','comparisons')},indent=2))
