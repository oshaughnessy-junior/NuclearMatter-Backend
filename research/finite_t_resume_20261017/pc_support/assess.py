from pathlib import Path
import json,math,hashlib,datetime
P=Path(__file__).resolve().parent
raw=json.loads((P/'CONVERGENCE_RAW.json').read_text());rows=raw['rows'];checks=[]
def add(name,error,gate):checks.append({'name':name,'error':error,'gate':gate,'pass':error<gate})
for E in (.1,.15):
 by={r['tag']:r for r in rows if r.get('E_MeV')==E}
 if len(by)!=5:raise ArithmeticError('Incomplete selected forward records')
 for component in ('log_j_s-1','log_true_chi_s-1'):
  for tag in ('coarse','inner','tail90','literal'):
   add(f'{E}_{component}_{tag}',abs(math.expm1(by[tag][component]-by['higher'][component])),3e-6)
 for tag,r in by.items():
  add(f'{E}_{tag}_strict_shell',r['max_shell_relative'],1e-10)
  add(f'{E}_{tag}_unsquared_delta',r['max_unsquared_delta_scaled'],1e-10)
  add(f'{E}_{tag}_four_conservation',r['max_four_conservation_scaled'],1e-10)
  add(f'{E}_{tag}_KMS_actual_products',abs(r['KMS_log_residual']),2e-9)
  if r['ordinary_j_s-1'] is not None or r['ordinary_true_chi_s-1'] is not None:raise ArithmeticError('Unexpected representation drift')
for row in json.loads((P/'PHASE_JACOBIAN_RAW.json').read_text()):
 if row['h']==.0003:add(f'{row["E"]}_absolute_phase_Jacobian',abs(row['relative_error']),3e-6)
for r in json.loads((P/'SEMI_INFINITE_RADIAL_RAW.json').read_text())['rows']:
 reference=next(x for x in rows if x['E_MeV']==r['E_MeV'] and x['tag']=='higher')
 for component in ('log_j_s-1','log_true_chi_s-1'):
  add(f'{r["E_MeV"]}_semi_infinite_N{r["rule"]["radial"]}_{component}',abs(math.expm1(r[component]-reference[component])),3e-6)
 add(f'{r["E_MeV"]}_semi_infinite_N{r["rule"]["radial"]}_shell',r['max_shell_relative'],1e-10)
 add(f'{r["E_MeV"]}_semi_infinite_N{r["rule"]["radial"]}_delta',r['max_unsquared_delta_scaled'],1e-10)
orig=json.loads((P/'ORIGINAL_FAILURE.json').read_text())
if orig['complete'] or orig['error']['message']!='Final positive-energy baryon shell unresolved':raise ArithmeticError('Failed exact original reproducer identity')
for old,expected in json.loads((P/'ORIGINAL_SOURCE_PIN.json').read_text()).items():
 if hashlib.sha256(Path(old).read_bytes()).hexdigest()!=expected:raise ArithmeticError('Published original mutated')
 if hashlib.sha256((P/'pr55_reference'/Path(old).name).read_bytes()).hexdigest()!=expected:raise ArithmeticError('Source-copy drift')
out={'status':'PASS_SELECTED_CONDITIONAL_SLICE' if all(c['pass'] for c in checks) else 'FAIL','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checks':checks,'whole_energy_or_transport_qualified':False,'independence':'Literal spinor vs invariant current and finite-difference absolute phase-Jacobian anchor; integrated phase oracle by DIFFERENT reviewer remains pending','forward_rows':[{k:r[k] for k in ('E_MeV','log_j_s-1','log_true_chi_s-1','threshold_initial_proton_energy_MeV')} for r in rows if r['tag']=='higher']}
(P/'ASSESSMENT.json').write_text(json.dumps(out,indent=2)+'\n');print(out['status'],len(checks),'checks')
if any(not c['pass'] for c in checks):raise SystemExit(1)
