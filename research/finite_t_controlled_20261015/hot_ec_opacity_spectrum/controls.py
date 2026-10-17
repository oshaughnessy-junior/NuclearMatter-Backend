"""Fixed-state guard and algebra checks, no broad-domain numerical claim."""
from pathlib import Path
import importlib.util,sys,json,math
P=Path(__file__).parent;sp=importlib.util.spec_from_file_location('hot_EC_spectrum_control_function',P/'spectrum.py');S=importlib.util.module_from_spec(sp);sys.modules[sp.name]=S;sp.loader.exec_module(S);checks=[]
def ck(n,ok):checks.append(dict(name=n,passed=bool(ok)))
def reject(name,fn):
 try:fn();ok=False
 except(ValueError,RuntimeError,ArithmeticError):ok=True
 ck(name,ok)
for E in[True,False,float('nan'),float('inf'),0.,-1.,60.01,'2',None]:reject('bad incident '+str(E),lambda E=E:S.integrate(E))
for kw in[dict(lepton_order=True),dict(lepton_order=3),dict(angle_order=False),dict(angle_order=3),dict(energy_order=True),dict(energy_order=7),dict(phi_order=False),dict(phi_order=3),dict(tail=float('nan')),dict(tail=29.9),dict(tail=100.1),dict(outer_tail=True),dict(outer_tail=float('inf')),dict(outer_tail=19.9),dict(budget_seconds=True),dict(budget_seconds=0),dict(budget_seconds=120.1)]:reject('bad controls '+str(kw),lambda kw=kw:S.integrate(5.,**kw))
b,meta=S.matched();reject('arbitrary unmatched bath',lambda:S.differential(S.C.Bath(b.T,b.mu_n,b.mu_p,b.mu_e+.001),5.,6.29,.1))
for en,el,c in[(True,6.,.1),(5.,S.C.ME,.1),(5.,float('nan'),.1),(5.,6.,float('nan')),(5.,6.,1.01)]:reject('bad differential '+str((en,el,c)),lambda en=en,el=el,c=c:S.differential(b,en,el,c))
for log in[-800.,800.,None,-3.]:
 r=S.represented(log);ck('ordinary/log convention '+str(log),(r['value']is None and not r['representable']and r['log']==log)if log in[-800.,800.]else(r['value']==0 and r['reason']=='empty_support')if log is None else r['value']>0 and r['representable'])
raw=json.loads((P/'results.json').read_text())
for z in raw['rows']:
 r=z['rules'][-1];j=r['j']['value'];chi=r['chi']['value'];fn=lambda f:j*(1-f)-chi*f
 ck('endpoint emissivity '+str(z['Enu']),fn(0.)==j);ck('true absorption at filled state '+str(z['Enu']),fn(1.)==-chi);ck('relaxation slope '+str(z['Enu']),abs((fn(.6)-fn(.2))/.4+j+chi)<2e-15*(j+chi))
ck('exact matched EOS metadata',meta==json.loads((P.parent/'hot_electron_capture_moments/MATCHED_STATE.json').read_text()))
# No ambiguous generic import aliases or sys.path are left changed by a fresh import.
path0=list(sys.path);old={k:sys.modules.get(k)for k in['current','vacuum_cc','lepton_reference']};sp=importlib.util.spec_from_file_location('hot_EC_spectrum_new_isolation_import',P/'spectrum.py');M=importlib.util.module_from_spec(sp);sys.modules[sp.name]=M;sp.loader.exec_module(M);ck('parent sys.path restored',sys.path==path0);ck('parent generic aliases restored',all(sys.modules.get(k)is v for k,v in old.items()))
result=dict(checks=checks,passed=sum(v['passed']for v in checks),failed=[v for v in checks if not v['passed']],scope='admission/algebra/representation only; no broad source-state domain or generic security sandbox');(P/'control_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=result['passed'],failed=result['failed']),indent=2))
