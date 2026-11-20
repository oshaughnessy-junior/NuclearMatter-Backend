"""Optional positive moment approximation; never returns a net source or root."""
from pathlib import Path
import sys,importlib.util,hashlib,json,numbers,math,numpy as np
P=Path(__file__).resolve().parent;sys.dont_write_bytecode=True
s=importlib.util.spec_from_file_location('optional_capture_fit_physical_reference_functions',P/'collect.py');M=importlib.util.module_from_spec(s);sys.modules[s.name]=M;s.loader.exec_module(M)
COEFFICIENT_SHA='6e81f38d4538732abc49462331c7277160ba220518747f43aca1b13421d71a12'
SOURCE_PIN_SHA='7fbc7c3f74f2f79ed46ce9a7c3bfba1376bb205f1658e76b7c54e5ac61f3385f'
QUALIFICATION_SHA='62f56d9127ef2daeec15911128c163e253044d32860b8200cf1b32d11a1642b1'
def _admitted_fit():
 """Admit the exact complete frozen qualification and its physical sources."""
 try:
  source_raw=(P/'SOURCE_PIN.json').read_bytes()
  if hashlib.sha256(source_raw).hexdigest()!=SOURCE_PIN_SHA:raise ValueError('source pin bytes')
  for name,digest in json.loads(source_raw).items():
   if hashlib.sha256((P.parent/name).read_bytes()).hexdigest()!=digest:raise ValueError('physical source bytes')
  lock_raw=(P/'REVISED_LOCK2.json').read_bytes()
  if hashlib.sha256(lock_raw).hexdigest()!=M.LOCK_SHA:raise ValueError('locked model bytes')
  # Parse precisely the bytes whose digest was checked; never hash one read
  # then consume a separately obtained report or coefficient document.
  coeff_raw=(P/'COEFFICIENT_FREEZE.json').read_bytes()
  if hashlib.sha256(coeff_raw).hexdigest()!=COEFFICIENT_SHA:raise ValueError('coefficient bytes')
  f=json.loads(coeff_raw)
  if f['lock_sha256']!=M.LOCK_SHA or f['rank']!=7 or f['outputs']!=['R_EC','Q_EC','R_PC','Q_PC']:raise ValueError('coefficient provenance')
  report_raw=(P/'HELDOUT_ASSESSMENT.json').read_bytes()
  if hashlib.sha256(report_raw).hexdigest()!=QUALIFICATION_SHA:raise ValueError('qualification bytes')
  q=json.loads(report_raw);lock=json.loads(lock_raw)
  names=[f'heldout{i} {name}'for i in range(8)for name in f['outputs']]
  if q['qualified']is not True or q['coefficient_sha256']!=COEFFICIENT_SHA or q['passed']!=32 or type(q['passed'])is not int or q['failed']!=[] or q['basis_retuned']is not False:raise ValueError('qualification summary')
  if len(q['checks'])!=32 or [c['name']for c in q['checks']]!=names or len(q['rows'])!=8:raise ValueError('complete qualification')
  for c in q['checks']:
   if c['passed']is not True or type(c['error'])not in(int,float)or not math.isfinite(c['error'])or type(c['gate'])not in(int,float)or c['gate']!=.01 or abs(c['error'])>.01:raise ValueError('passing heldout gate')
  for i,row in enumerate(q['rows']):
   if type(row['index'])is not int or row['index']!=i or row['state']!=lock['heldout'][i]:raise ValueError('heldout state/index')
   raw=(P/f'heldout_{i}.json').read_bytes()
   if hashlib.sha256(raw).hexdigest()!=row['raw_sha256']:raise ValueError('physical heldout source bytes')
   r=json.loads(raw)
   if r['qualified']is not True or r['state']!=row['state']or r['group']!='heldout'or r['index']!=i or r['sourcepins']!=M.PINS or row['actual']!=r['refined']['values']:raise ValueError('physical heldout provenance')
   if len(r['checks'])!=8 or any(c['passed']is not True for c in r['checks']):raise ValueError('physical source gates')
  if len(f['training_files'])!=11:raise ValueError('training source completeness')
  for name,digest in f['training_files'].items():
   if hashlib.sha256((P/name).read_bytes()).hexdigest()!=digest:raise ValueError('training source bytes')
  return f
 except (OSError,ValueError,KeyError,TypeError,IndexError)as exc:
  raise RuntimeError('Frozen complete passing qualification/source/coefficient evidence required')from exc
def approximate_moments(T,nB,Ye):
 if not all(isinstance(x,numbers.Real)and not isinstance(x,(bool,np.bool_))and math.isfinite(x)for x in[T,nB,Ye])or not 1<=T<=3 or not 1e-7<=nB<=1e-6 or not .2<=Ye<=.5:raise ValueError('Selected box and finite real scalar inputs required; no extrapolation')
 f=_admitted_fit()
 b,st=M.EC.matched(float(T),float(nB),float(Ye));eta=b.mu_e/T
 if abs(eta)>3 or max((b.mu_n-M.EC.MN)/T,(b.mu_p-M.EC.MP)/T)>-5:raise ValueError('Actual EOS outside retained event-kernel admission; rectangle alone insufficient')
 base=[]
 for ch,target in [('electron_capture',nB*Ye),('positron_capture',nB*(1-Ye))]:base.extend((target*M.S.target_moments(T,b.mu_e,ch,q=M.Q)).tolist())
 log=np.array(M.basis(T,nB,eta))@np.array(f['coefficients']);values=np.array(base)*np.exp(log)
 if not np.all(np.isfinite(values))or np.any(values<=0):raise ArithmeticError('Positive finite moment required')
 return dict(moments=dict(zip(f['outputs'],map(float,values))),baseline=dict(zip(f['outputs'],base)),log_corrections=log.tolist(),state=dict(T=T,nB=nB,Ye=Ye,mu_e=b.mu_e,eta_e=eta),model='OPTIONAL positive EC/PC moment approximation; detailed physical phase integrals remain reference/default; no net source or root',qualification='Author locked heldouts only; different independent review required separately',coefficient_sha256=COEFFICIENT_SHA,error_contract='Only locked heldouts sampled; no uniform domain-error/derivative/current/EOS-family/physical-uncertainty guarantee')
