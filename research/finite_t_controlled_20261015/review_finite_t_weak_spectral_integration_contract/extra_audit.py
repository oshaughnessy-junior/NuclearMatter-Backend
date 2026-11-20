from pathlib import Path
import json,math
P=Path(__file__).resolve().parent;R=P.parent;A=R/'finite_t_weak_spectral_integration_contract';S=json.loads((A/'SAVED_EVIDENCE.json').read_text());checks=[]
def ck(n,b):checks.append(dict(name=n,passed=bool(b)))
parts={'EC':[f'hot_ec_opacity_spectrum/total_fine_part{i}.json'for i in range(2)],'PC':[f'hot_pc_opacity_spectrum/total_fine_{i}.json'for i in range(4)],'NEUTRON_DECAY':[f'hot_beta_opacity_spectrum/total_fine_{i}.json'for i in range(2)]}
for row in S['totals']:
 c=row['contract'];ch=row['channel'];keys=['parent_order','cone_order','electron_order','phi_order','tail']if ch=='NEUTRON_DECAY'else['lepton_order','angle_order','energy_order','phi_order','tail','outer_tail'];wanted=dict(zip(keys,c['controls']))
 for name in parts[ch]:
  for r in json.loads((R/name).read_text())['rows']:ck(ch+'exactactualcontrols '+str(r['index']),r['spectrum']['controls']==wanted)
# An abstract positive integration panel: NOT a new physical-rate evaluation.
lw,li=-750.,100.;width=math.exp(lw);integrand=math.exp(li);contribution=math.exp(lw+li)
ck('counterexample width ordinaryunderflow',width==0.);ck('counterexample integrand ordinaryrepresented',math.isfinite(integrand)and integrand>0);ck('counterexample trueweightedcontribution ordinaryrepresented',math.isfinite(contribution)and contribution>0)
text=(A/'CONTRACT.md').read_text();ck('REQUIRED fullweightedcontribution proof explicitly required insteadwidthonly','proven unrepresentable-width skip'not in text)
out=dict(kind='DESIGNcontrolchecks+abstractlogpanelcounterexample ONLY',passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],checks=checks,witness=dict(log_width=lw,ordinary_width=width,log_integrand=li,ordinary_integrand=integrand,log_weighted_contribution=lw+li,ordinary_weighted_contribution=contribution,interpretation='Width-only underflow/collapse cannot justify discarding a weightedpositive phasepanel; not actual newweakrate'),new_physical_calls=0)
(P/'EXTRA_RESULTS.json').write_text(json.dumps(out,indent=2)+'\n');print('EXTRA',out['passed'],'FAIL',len(out['failed']));print(out['failed'])
