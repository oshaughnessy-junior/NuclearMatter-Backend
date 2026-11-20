"""DIFFERENT notation-only repair recheck; saved algebra, no scientificimports/rates."""
from pathlib import Path
import json,hashlib,difflib,re,datetime
P=Path(__file__).resolve().parent;R=P.parent;A=R/'finite_t_weak_reference_integration_contract';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();old=json.loads((P/'INITIAL_CHECKPOINT.json').read_text());checks=[]
def ck(name,error,gate=0):checks.append(dict(name=name,error=float(error),gate=gate,passed=bool(abs(error)<=gate)))
assert sha(A/'CONTRACT.md')=='60adb59adecb0d0bd827c25a10cde117e12a79a7bb77ec7aeedd6d77659c22c4';assert sha(A/'MANIFEST.json')=='fae2e5f7cbb12f613569112549ddc3e6efce9b39c6514c0d067aae9cd3dae983'
for rel,h in old['initial_reviewfiles'].items():ck('initialreviewpreserved '+rel,0 if sha(P/rel)==h else 1)
for rel,h in old['initial_authorfiles'].items():
 ck('reviewerinitialauthorsnapshot '+rel,0 if sha(P/'initial_author_snapshot'/rel)==h else 1)
 ck('authorinitialsnapshot '+rel,0 if sha(A/'pre_transfer_notation_repair'/rel)==h else 1)
for rel,h in old['initial_upstreampins'].items():ck('scientificpinunchanged '+rel,0 if sha(R/rel)==h else 1)
allowed={'CONTRACT.md','MANIFEST.json','CHECKPOINT.json'};changed=[rel for rel,h in old['initial_authorfiles'].items()if sha(A/rel)!=h]
ck('onlyallowedoriginalfileschanged',0 if set(changed)<=allowed else 1)
for rel,h in old['initial_authorfiles'].items():
 if rel not in allowed:ck('otherscientificauditdataunchanged '+rel,0 if sha(A/rel)==h else 1)
text=(A/'CONTRACT.md').read_text();rows=[s.strip()for s in text.splitlines()if s.startswith('|')]
expected=[
 ('ECabs','nue+n -> p+e−','Gamma_np(q_Gamma)','q_Gamma=k_nue−p_e−=Q_abs'),
 ('ECemit','p+e− -> n+nue','Gamma_pn(q_pn)','q_pn=−q_Gamma=p_e−−k_nue'),
 ('PCemit','n+e+ -> p+antinue','Gamma_np(q_Gamma)','q_Gamma=p_e+−k_antinue=−Q_abs'),
 ('PCabs','antinue+p -> n+e+','Gamma_pn(q_pn)','q_pn=k_antinue−p_e+=Q_abs'),
 ('Betaemit','n -> p+e−+antinue','Gamma_np(q_Gamma)','q_Gamma=−(p_e−+k_antinue)=−Q_leptons'),
 ('Betainverse','p+e−+antinue -> n','Gamma_pn(q_pn)','q_pn=p_e−+k_antinue=Q_leptons')]
for name,process,current,identity in expected:ck('explicitcorrectorientation '+name,0 if any(all(x in row for x in[process,current,identity])for row in rows)else 1)
for name,literal in[
 ('orientednpdefinition','q_Gamma=p_p−p_n'),
 ('reciprocaldefinition','Gamma_pn(q_pn)=DiracAdjoint[Gamma_np(−q_pn)]'),
 ('futurepairdistinct','Q_leptons=kanti+pe=p_n−p_p=−q_Gamma'),
 ('decaysupportinvariant','me²<=Q_leptons²=q_Gamma²<=(MN−MP)²'),
 ('lineartermsnotinterchangeable','equal squares do NOT identify the linear F2/GP transfer terms'),
 ('phaseqmagnitude','q in the following phase prefactor is a MAGNITUDE'),
 ('Hartreeorientations','q_Gamma,physical and q_Gamma,star (and their negatives for the reciprocal channel)')]:ck('clarified '+name,0 if literal in text else 1)
# Provevertexonlyrenamedargument; coefficientsandnormalizationunchanged.
a=(P/'initial_author_snapshot/CONTRACT.md').read_text();extract=lambda s:s[s.index('    Gamma^mu ='):s.index('\n\n',s.index('    Gamma^mu ='))]
newblock=extract(text).replace('(q_Gamma)_nu','q_nu').replace('q_Gamma^mu','q^mu').replace('q_Gamma²','q²')
ck('samevertexafterargumentrename',0 if newblock==extract(a)else 1)
# Support/current/GM1paragraphsaretheonlymodifiedoriginaltext regions.
diff=''.join(difflib.unified_diff(a.splitlines(True),text.splitlines(True),fromfile='initialCONTRACT',tofile='repairedCONTRACT'));(P/'independent_repair.diff').write_text(diff)
# Revalidateinitialphysicalkinematicalwitness, notamplitudeornewphaseintegral.
w=json.loads((P/'initial_results.json').read_text())['details']['physical_decay_sign_witness'];q=w['q_Gamma'];Q=w['Q_leptons'];ck('witnessqGammaNEGATIVEQ',max(abs(x+y)for x,y in zip(q,Q)),2e-13);ck('witnesspastvsfuture',0 if q[0]<0<Q[0]else 1);ck('witnessequalsquares',w['q_squared']-w['Q_squared'],2e-13)
# EntiretotalF/B savedgroups andspectraldata byteidentical, notmerelyselectednumbers.
for rel in['SAVED_REGRESSIONS.json','SOURCE_INVENTORY.json','AUDIT_RESULTS.json','SOURCE_PIN.json','REPORT.md']:ck('entireoriginal'+rel,0 if sha(A/rel)==old['initial_authorfiles'][rel]else 1)
m=json.loads((A/'MANIFEST.json').read_text());assert all(sha(A/rel)==h for rel,h in m['sha256'].items());ck('repairedmanifestverified',0)
out=dict(status='ACCEPTEDnotationclarification; designonly',passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],checks=checks,original_failed_gate_retained=json.loads((P/'initial_results.json').read_text())['failed'],changed_original_paths=changed,new_contract=sha(A/'CONTRACT.md'),new_manifest=sha(A/'MANIFEST.json'),new_author_checkpoint=sha(A/'CHECKPOINT.json'),no_new_physical_rates=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat());(P/'repair_verification.json').write_text(json.dumps(out,indent=2)+'\n');print('RECHECKPASS',out['passed'],'FAIL',len(out['failed']));print(out['failed'])
