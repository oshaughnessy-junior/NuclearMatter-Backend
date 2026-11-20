from pathlib import Path
import json,hashlib,math,re,difflib
P=Path(__file__).resolve().parent;R=P.parent;A=R/'finite_t_weak_spectral_integration_contract';h=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();checks=[]
def ck(n,b):checks.append(dict(name=n,passed=bool(b)))
for n,v in{'CONTRACT.md':'5fa8a528cba8022a05aac8e821afdb07ecda1baaa260a1c59eb5997718af0340','MANIFEST.json':'deb946afc6896dfd00ddfacbdea546e30e3acbb8607a3adb4dbeaf108b5060a8','WEIGHTED_PANEL_REPAIR.md':'a776f05613ad280b46a7c95974e20edccf219e25254dd5e65eca7153cedca7f1'}.items():ck('repairedidentity '+n,h(A/n)==v)
M=json.loads((A/'MANIFEST.json').read_text())
for n,v in M['sha256'].items():ck('repairedownmanifest '+n,h(A/n)==v)
for n,v in json.loads((P/'INITIAL_MANIFEST.json').read_text())['files'].items():ck('initialreviewunchanged '+n,h(P/n)==v)
old=(P/'INITIAL_AUTHOR_CONTRACT.md').read_text();new=(A/'CONTRACT.md').read_text()
oldparas=old.split('\n\n');newparas=new.split('\n\n');remove=lambda paras:[p for p in paras if not p.startswith('Outer El integration resolves')and not p.startswith('Any optional panel omission requires')]
ck('ONLYpanelguardprosechanged',remove(oldparas)==remove(newparas));ck('oldwidthloopholepreserved','proven unrepresentable-width skip'in old);ck('widthonlyloopholeremoved','proven unrepresentable-width skip'not in new)
for s in['derived alternate coordinate or an explicit unresolved-panel exception','ordinary-output underflow ALONE NEVER','COMPLETE weighted contribution to BOTH j AND chi','current amplitude, all FD factors','invariant measures/Jacobians/angular factors','target coefficient/reconstruction units','Predeclare and independently validate','Retain log bounds','certified positive lower bounds','relative/log budgets','log1p(U/L)','absolute/gross source/energy budgets','not a bound on that complete contribution','Do not substitute an upper bound','only resolved positive contribution','finite-log semantics','final error/completeness diagnostic']:
 ck('requiredrepairpolicy '+s,s in new)
# Original abstract witness and analyticlogerrorbound arithmetic ONLY.
w=json.loads((P/'EXTRA_RESULTS.json').read_text())['witness'];ck('oldcounterexampleremainsrepresented',math.exp(w['log_width'])==0 and math.exp(w['log_weighted_contribution'])>0)
for L,U in[(1.,.01),(2.,3.),(.125,.375)]:
 for fraction in[0.,.2,.9,1.]:
  actual=math.log1p(fraction*U/L);upper=math.log1p(U/L);ck('certifiedlogerrorinequality '+str((L,U,fraction)),actual<=upper)
S=json.loads((A/'SOURCE_PIN.json').read_text())['sha256'];ck('95pinsremain95',len(S)==95)
for n,v in S.items():ck('scientific/evidenceunchanged '+n,h(R/n)==v)
for n,v in json.loads((P/'INITIAL_AUTHOR_MANIFEST.json').read_text())['sha256'].items():
 if n!='CONTRACT.md':ck('oldauthornonscientificunchanged '+n,h(A/n)==v)
# Whole initialauthoredfiles snapshot is checked againstoriginal source identities.
originalpin=json.loads((A/'PRE_WEIGHTED_PANEL_REPAIR_PIN.json').read_text());(P/'PINNED_REPAIR_PROVENANCE.json').write_text(json.dumps(originalpin,indent=2)+'\n')
for n,v in originalpin['initial_author_files'].items():ck('preservedinitialauthorsnapshot '+n,h(A/'pre_weighted_panel_repair'/n)==v)
for n,v in originalpin['initial_review_files'].items():ck('authorpinnedinitialreview '+n,h(P/n)==v)
(P/'CONTRACT_DIFF.txt').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='initial_frozen_contract',tofile='repaired_frozen_contract')))
out=dict(passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],checks=checks,new_physical_calls=0,scientific_module_imports=0,scope='text-onlyweightedpaneladmissionrepair, NOT implementationvalidation',newcontract=h(A/'CONTRACT.md'),newmanifest=h(A/'MANIFEST.json'))
(P/'RECHECK_RESULTS.json').write_text(json.dumps(out,indent=2)+'\n');print('RECHECK',out['passed'],'FAIL',len(out['failed']));print(out['failed'])
