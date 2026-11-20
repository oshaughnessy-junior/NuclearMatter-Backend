"""Static saved-data and algebra audit; imports NO scientific campaign module."""
from pathlib import Path
import json,hashlib,math,shutil,datetime
import numpy as np
P=Path(__file__).resolve().parent;R=P.parent;A=R/'finite_t_weak_reference_integration_contract';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();checks=[];details={}
def ck(name,error,gate):checks.append(dict(name=name,error=float(error),gate=gate,passed=bool(abs(error)<=gate)))
assert sha(A/'CONTRACT.md')=='bb836077dbca372d0e231958d37788cc66b24266811522ee91751cae552fb370';assert sha(A/'MANIFEST.json')=='eef214edf29d993a1b87323a1cfc402e21396b20b7f0d607639c0a7617415e6e'
initial={str(p.relative_to(A)):sha(p)for p in A.rglob('*')if p.is_file()}
for rel,h in initial.items():
 q=P/'initial_author_snapshot'/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(A/rel,q);assert sha(q)==h
pins=json.loads((A/'SOURCE_PIN.json').read_text())['sha256']
for rel,h in pins.items():ck('upstreampin '+rel,0 if sha(R/rel)==h else 1,0)
manifest=json.loads((A/'MANIFEST.json').read_text())['sha256']
for rel,h in manifest.items():ck('frozenauthor '+rel,0 if sha(A/rel)==h else 1,0)
saved=json.loads((A/'SAVED_REGRESSIONS.json').read_text())
def pointer(src,ptr):
 for key in ptr.strip('/').split('/'):src=src[int(key)]if isinstance(src,list)else src[key]
 return src
for i,r in enumerate(saved['spectra']):
 ck('savedsourcehash '+str(i),0 if sha(R/r['source'])==r['source_sha256']else 1,0);q=pointer(json.loads((R/r['source']).read_text()),r['json_pointer'])
 for key,target in [('j','j'),('true_chi','chi'),('relaxation','relaxation')]:ck('savedexact '+str(i)+key,0 if r[key]==q[target]else 1,0)
 b=r['bath'];mu=b['mu_p']+b['mu_e']-b['mu_n'];mu=mu if r['channel']=='EC'else -mu;logratio=r['j']['log']-r['true_chi']['log'];ck('savedKMS '+str(i),logratio+(r['E_MeV']-mu)/b['T'],2e-8);ck('savedtruechi/relax '+str(i),r['relaxation']/(r['j']['value']+r['true_chi']['value'])-1,2e-14)
for i,r in enumerate(saved['totals']):
 ck('savedtotalsourcehash '+str(i),0 if sha(R/r['source'])==r['source_sha256']else 1,0);q=pointer(json.loads((R/r['source']).read_text()),r['json_pointer']);ck('savedexactF/Btotals '+str(i),0 if q==r['total']else 1,0)
 for label,v in r['total'].items():
  if label=='empty':ck('emptyinverseexactzero '+str(i),max(abs(x)for x in v['B']),0)
  if label=='LTE':
   for j in range(2):ck('LTEpositivebalanced '+str(i)+str(j),v['F'][j]/v['B'][j]-1,2e-8);ck('LTEbranchpositive '+str(i)+str(j),0 if min(v['F'][j],v['B'][j])>0 else 1,0)
# Independent integer reactionledgers order(n,p,e_net,nue,antinue).
reactions={'EC':[1,-1,-1,1,0],'PC':[-1,1,1,0,1],'NEUTRON_DECAY':[-1,1,1,0,1]}
for ch,v in reactions.items():
 ck('baryon '+ch,v[0]+v[1],0);ck('charge '+ch,v[1]-v[2],0);ck('electronLe '+ch,v[2]+v[3]-v[4],0)
# Masterspin/phase factors yield acceptedavg4 differentialprefactor.
phase=(1/(2*16*8*math.pi**4))*4*(2*math.pi);ck('avg4masterphaseprefactor',phase/(1/(32*math.pi**3))-1,2e-15)
HC=197.3269804;MN=939.56542052;MP=938.27208816;ME=.51099895069;C=2.99792458e10
thr=(MN+ME-MP)*(MN+ME+MP)/(2*MP);end=(MN-MP-ME)*(MN+MP+ME)/(2*MN);ck('targetrestthreshold',thr-1.8060662080187608,2e-10);ck('neutronrestendpoint',end-.7820077025790402,2e-10)
ck('lengthcm-to-km',1e5/C/(1/(C*1e-5))-1,2e-15)
# Exactphysicalfree-neutrondecayevent to disambiguate currenttransfer vs pairQ.
Ee=1.;co=.23;pe=math.sqrt(Ee*Ee-ME*ME);eemax=((MN-MP)*(MN+MP)+ME*ME)/(2*MN);enu=MN*(eemax-Ee)/(MN-Ee+pe*co);pn=np.array([MN,0.,0.,0.]);l=np.array([Ee,0.,0.,pe]);k=np.array([enu,enu*math.sqrt(1-co*co),0.,enu*co]);pp=pn-l-k;qg=pp-pn;Q=l+k;dot=lambda v:v[0]*v[0]-sum(v[1:]*v[1:]);ck('decayonshellproton',dot(pp)/(MP*MP)-1,2e-13);ck('qGammaNEGATIVEQpair',np.linalg.norm(qg+Q),2e-13);ck('sameq2not_samevector',dot(qg)-dot(Q),2e-13)
ck('pastcurrentvsfuturepair',0 if qg[0]<0 and Q[0]>0 and dot(Q)>ME*ME else 1,0)
details['physical_decay_sign_witness']=dict(pn=pn.tolist(),pp=pp.tolist(),electron=l.tolist(),antinu=k.tolist(),q_Gamma=qg.tolist(),Q_leptons=Q.tolist(),q_squared=dot(qg),Q_squared=dot(Q),no_matrixelement_or_rate_evaluated=True)
# Initial REQUIREDclarification; actualfrozencampaignamplitudeisalreadycorrect.
ck('CONTRACTexplicitlydistinguishesdecayqGammaandfutureQ',1,0)
text=(A/'CONTRACT.md').read_text();details['initial_ambiguous_literals']=[s for s in text.splitlines()if 'For n->p baryon current with q='in s or 'Decay q='in s];details['source_beta_literal']='relativistic_crossed_contract/beta.py: q=pp-pn; reciprocal adj(V), unchangedsource'
assert {str(p.relative_to(A)):sha(p)for p in A.rglob('*')if p.is_file()}==initial;assert all(sha(R/k)==h for k,h in pins.items())
out=dict(status='INITIALREVIEW_REQUIRED_TRANSFER_NOTATION_CLARIFICATION',passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],checks=checks,details=details,authorfiles=initial,upstreampins=pins,no_new_physical_rate_or_module_import=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat());(P/'initial_results.json').write_text(json.dumps(out,indent=2)+'\n');print('PASS',out['passed'],'CLARIFICATIONFAIL',len(out['failed']))
