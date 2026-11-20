"""Read-only static provenance/saved-row audit; no scientific module imports."""
from pathlib import Path
import ast, hashlib, json, datetime
P=Path(__file__).resolve().parent
R=P.parent
files='''hot_electron_capture_moments/electron_capture.py
hot_electron_capture_moments/RESEARCH_NOTE.md
review_hot_electron_capture/REVIEW.md
review_hot_electron_capture/CHECKPOINT.json
review_pc_moments/reference.py
review_pc_moments/REVIEW.md
review_pc_moments/MANIFEST.json
positron_capture/pc_kernel.py
positron_capture_moments/moments.py
hot_beta_moments/hot_moments.py
hot_beta_moments/results.json
hot_beta_moments/RESEARCH_NOTE.md
review_hot_beta/REVIEW.md
hot_ec_opacity_spectrum/spectrum.py
hot_ec_opacity_spectrum/total.py
hot_ec_opacity_spectrum/results.json
hot_ec_opacity_spectrum/total_results.json
hot_ec_opacity_spectrum/RESEARCH_NOTE.md
review_hot_ec_opacity_spectrum/REVIEW.md
hot_pc_opacity_spectrum/spectrum.py
hot_pc_opacity_spectrum/total.py
hot_pc_opacity_spectrum/results.json
hot_pc_opacity_spectrum/total_results.json
hot_pc_opacity_spectrum/RESEARCH_NOTE.md
review_hot_pc_opacity_spectrum/REVIEW.md
review_hot_pc_assembly_repair/REVIEW.md
hot_beta_opacity_spectrum/spectrum.py
hot_beta_opacity_spectrum/total.py
hot_beta_opacity_spectrum/pilot_results.json
hot_beta_opacity_spectrum/total_results.json
hot_beta_opacity_spectrum/RESEARCH_NOTE.md
review_hot_beta_opacity_spectrum/REVIEW.md
muon_full_current/current.py
muon_full_current/SOURCES.json
muon_vacuum/vacuum_cc.py
relativistic_capture_oracle/events.py
relativistic_tensor/tensor.py
relativistic_crossed_contract/beta.py
relativistic_crossed_thermal/kernel.py
relativistic_free_eos/free_baryons.py
thermodynamics/lepton_reference.py
transparent_capture_balance/balance.py
transparent_balance_grid/grid.py
three_source_cooling_element/worker.py
isolated_microphysics/client.py
isolated_microphysics/protocol.py
isolated_microphysics/worker_entry.py
installed_free_matter_binding/matter.py
installed_weak_density_element/element.py
free_capture_moment_fit/approximation.py
free_capture_moment_fit/collect.py
free_capture_moment_fit/RESEARCH_NOTE.md
free_capture_moment_fit/COEFFICIENT_FREEZE.json
free_capture_moment_fit/REVISED_LOCK2.json
review_free_capture_moment_fit/REVIEW.md
review_free_capture_moment_fit_repair/REVIEW.md
allowed_rate_assessment/RESEARCH_NOTE.md
cold_capture/RESEARCH_NOTE.md
emission_only/RESEARCH_NOTE.md
finite_t_rmf_integration_contract/CONTRACT.md'''.splitlines()
# Missing source names are explicit audit diagnostics, not silently guessed imports.
missing=[x for x in files if not (R/x).is_file()]
if missing:
 raise FileNotFoundError('Audit inventory paths require explicit correction: '+repr(missing))
raw={x:(R/x).read_bytes() for x in files}
sha=lambda b:hashlib.sha256(b).hexdigest()
pins={x:sha(b) for x,b in raw.items()}
inv=[]
for rel,b in raw.items():
 if not rel.endswith('.py'):continue
 tree=ast.parse(b.decode())
 def calls(z):return any(isinstance(q,ast.Call)for q in ast.walk(z))
 inv.append(dict(path=rel,sha256=pins[rel],imports=[ast.unparse(z)for z in ast.walk(tree)if isinstance(z,(ast.Import,ast.ImportFrom))],public_functions=[dict(name=z.name,arguments=ast.unparse(z.args))for z in tree.body if isinstance(z,ast.FunctionDef)and not z.name.startswith('_')],classes=[z.name for z in tree.body if isinstance(z,ast.ClassDef)],top_level_call_bearing_statements=[ast.unparse(z)[:1200]for z in tree.body if not isinstance(z,(ast.FunctionDef,ast.ClassDef))and calls(z)]))
rows=[]
for channel,rel,energykey in [('EC','hot_ec_opacity_spectrum/results.json','Enu'),('PC','hot_pc_opacity_spectrum/results.json','E'),('NEUTRON_DECAY','hot_beta_opacity_spectrum/pilot_results.json','E')]:
 data=json.loads(raw[rel])
 for i,row in enumerate(data['rows']):
  rule=row['rules'][-1]
  rows.append(dict(channel=channel,source=rel,json_pointer=f'/rows/{i}/rules/{len(row["rules"])-1}',source_sha256=pins[rel],E_MeV=row[energykey],bath=rule['bath'],j=rule['j'],true_chi=rule['chi'],relaxation=rule['relaxation'],controls=rule.get('controls'),interpretation='SAVED physical coefficients, not recomputed'))
totals=[]
for channel,rel in [('EC','hot_ec_opacity_spectrum/total_results.json'),('PC','hot_pc_opacity_spectrum/total_results.json'),('NEUTRON_DECAY','hot_beta_opacity_spectrum/total_results.json')]:
 d=json.loads(raw[rel]);g=d['groups']['fine'];totals.append(dict(channel=channel,source=rel,source_sha256=pins[rel],json_pointer='/groups/fine/total',total=g['total'],contract=g['contract'],units=['fm^-3 s^-1','MeV fm^-3 s^-1']))
# No physical or derivative testing: exact snapshots and saved KMS arithmetic only.
checks=[]
for i,row in enumerate(rows):
 b=row['bath'];mu=b['mu_p']+b['mu_e']-b['mu_n'];mu=mu if row['channel']=='EC'else-mu
 residual=(row['j']['log']-row['true_chi']['log'])+(row['E_MeV']-mu)/b['T']
 checks.append(dict(name=f'saved_{row["channel"]}_{i}_KMS_arithmetic',passed=abs(residual)<1e-10,residual=residual))
for rel,b in raw.items():checks.append(dict(name='source_unchanged:'+rel,passed=(R/rel).read_bytes()==b))
write=lambda n,d:(P/n).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
write('SOURCE_PIN.json',dict(kind='read-only audit selected sources and accepted reviews; not an executed import-tree certificate',sha256=pins))
write('SOURCE_INVENTORY.json',dict(method='stdlib AST only; scientific modules NOT imported',modules=inv))
write('SAVED_REGRESSIONS.json',dict(kind='literal immutable saved selected results, no fresh physics qualification',spectra=rows,totals=totals))
write('AUDIT_RESULTS.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),new_physical_evaluations=0,checks=checks,passed=sum(c['passed']for c in checks),failed=sum(not c['passed']for c in checks)))
print(json.dumps(dict(source_files=len(pins),static_modules=len(inv),saved_spectra=len(rows),saved_total_groups=len(totals),checks=len(checks),failed=sum(not c['passed']for c in checks))))
