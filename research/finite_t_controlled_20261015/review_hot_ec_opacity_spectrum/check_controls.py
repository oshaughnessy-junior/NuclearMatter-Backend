from pathlib import Path
import importlib.util,sys,json,math
P=Path(__file__).resolve().parent;sys.dont_write_bytecode=True
sp=importlib.util.spec_from_file_location('hotEC_review_control_functions',P.parent/'hot_ec_opacity_spectrum/spectrum.py');S=importlib.util.module_from_spec(sp);sys.modules[sp.name]=S;sp.loader.exec_module(S)
def run():
 checks=[];b,_=S.matched()
 def rejects(name,f):
  try:f();ok=False
  except (ValueError,ArithmeticError):ok=True
  checks.append(dict(name=name,passed=ok))
 for E in [float('nan'),float('inf'),True,0.,-1.,60.01]:rejects('invalidincident '+repr(E),lambda E=E:S.integrate(E))
 for key,value in [('lepton_order',True),('angle_order',7.5),('energy_order',7),('phi_order',True),('tail',float('nan')),('outer_tail',float('inf')),('budget_seconds',float('nan'))]:rejects('invalidcontrol '+key,lambda key=key,value=value:S.integrate(7.3,**{key:value}))
 wrong=S.C.Bath(b.T,b.mu_n,b.mu_p,b.mu_e+.01);rejects('rejectchangedmatchedstate',lambda:S.differential(wrong,7.3,8.58,-.47))
 for log,name,value,reason in [(None,'empty',0.,'empty_support'),(-1000.,'underflow',None,'underflow'),(1000.,'overflow',None,'overflow'),(0.,'positiveunit',1.,None)]:
  r=S.represented(log);checks.append(dict(name='representation '+name,passed=r['value']==value and r['reason']==reason and r['log']==log))
 for log in [float('nan'),float('inf')]:rejects('rejectnonfinitelog '+str(log),lambda log=log:S.represented(log))
 out=dict(checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']]);(P/'controls_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':run()
