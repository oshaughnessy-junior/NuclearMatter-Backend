"""Optional positive moment approximation; never returns a net source or root."""
from pathlib import Path
import sys,importlib.util,hashlib,json,numbers,math,numpy as np
P=Path(__file__).resolve().parent;sys.dont_write_bytecode=True
s=importlib.util.spec_from_file_location('optional_capture_fit_physical_reference_functions',P/'collect.py');M=importlib.util.module_from_spec(s);sys.modules[s.name]=M;s.loader.exec_module(M)
COEFFICIENT_SHA='6e81f38d4538732abc49462331c7277160ba220518747f43aca1b13421d71a12'
def approximate_moments(T,nB,Ye):
 if not all(isinstance(x,numbers.Real)and not isinstance(x,(bool,np.bool_))and math.isfinite(x)for x in[T,nB,Ye])or not 1<=T<=3 or not 1e-7<=nB<=1e-6 or not .2<=Ye<=.5:raise ValueError('Selected box and finite real scalar inputs required; no extrapolation')
 M.verify()
 if hashlib.sha256((P/'COEFFICIENT_FREEZE.json').read_bytes()).hexdigest()!=COEFFICIENT_SHA:raise RuntimeError('Frozen coefficients changed')
 qualification=P/'HELDOUT_ASSESSMENT.json'
 if not qualification.exists():raise RuntimeError('Candidate lacks locked heldout qualification')
 q=json.loads(qualification.read_text())
 if q.get('qualified')is not True or q.get('coefficient_sha256')!=COEFFICIENT_SHA:raise RuntimeError('Candidate not qualified by matching locked heldout gates; no approximation admitted')
 b,st=M.EC.matched(float(T),float(nB),float(Ye));eta=b.mu_e/T
 if abs(eta)>3 or max((b.mu_n-M.EC.MN)/T,(b.mu_p-M.EC.MP)/T)>-5:raise ValueError('Actual EOS outside retained event-kernel admission; rectangle alone insufficient')
 f=json.loads((P/'COEFFICIENT_FREEZE.json').read_text());base=[]
 for ch,target in [('electron_capture',nB*Ye),('positron_capture',nB*(1-Ye))]:base.extend((target*M.S.target_moments(T,b.mu_e,ch,q=M.Q)).tolist())
 log=np.array(M.basis(T,nB,eta))@np.array(f['coefficients']);values=np.array(base)*np.exp(log)
 if not np.all(np.isfinite(values))or np.any(values<=0):raise ArithmeticError('Positive finite moment required')
 return dict(moments=dict(zip(f['outputs'],map(float,values))),baseline=dict(zip(f['outputs'],base)),log_corrections=log.tolist(),state=dict(T=T,nB=nB,Ye=Ye,mu_e=b.mu_e,eta_e=eta),model='OPTIONAL positive EC/PC moment approximation; detailed physical phase integrals remain reference/default; no net source or root',qualification='Author locked heldouts only; different independent review required separately',coefficient_sha256=hashlib.sha256((P/'COEFFICIENT_FREEZE.json').read_bytes()).hexdigest(),error_contract='Only locked heldouts sampled; no uniform domain-error/derivative/current/EOS-family/physical-uncertainty guarantee')
