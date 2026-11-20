from pathlib import Path
import sys,json,time,math,hashlib
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'crossed_thermal_moments'));import spectrum_anchor as S
T=.055;nB=2.3e-12;Ye=.31;E=.365;b,m=S.M.K.matched_npe(T,nB,Ye);start=time.monotonic();runs=[S.conditional(b,E,order=o,parent_order=p,phi_order=f)for o,p,f in[(10,12,8),(16,18,12)]]
(HERE/'spectrum_partial.json').write_text(json.dumps(dict(runs=runs,material=m),indent=2)+'\n')
tensor=S.M.K.spectral(b,E,angle_order=16,lepton_order=20,widths=12,energy_order=24,phi_order=4,tail=80,budget_seconds=45);checks=[]
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
for name in ['j','chi']:
 gate('new conditional spectrum refinement '+name,math.expm1(runs[0][name]['log']-runs[1][name]['log']),3e-6);gate('new conditional vs tensor '+name,math.expm1(runs[1][name]['log']-tensor[name]['log']),3e-6)
# Independently derive and differentiate the conditional orientation delta.
gamma=1.003;v=math.sqrt(1-1/gamma**2);erest=.41;c=.27;h=1e-5;fp=gamma*erest*(1+v*(c+h));fm=gamma*erest*(1+v*(c-h));jac=1/(2*gamma*v*erest);gate('isotropic-cone original delta derivative',jac*2*(fp-fm)/(2*h)-1,3e-10);gate('one-helicity DOS',runs[-1]['DOS']/(E*E/(2*math.pi**2*S.M.HC**3))-1,0.)
(HERE/'spectrum_results.json').write_text(json.dumps(dict(passed=all(c['passed']for c in checks),checks=checks,runs=runs,tensor=tensor,runtime_seconds=time.monotonic()-start,scope='New selected .365MeV spectral check only, no full-Enu tensor integral'),indent=2)+'\n');print(len(checks),all(c['passed']for c in checks),time.monotonic()-start)
