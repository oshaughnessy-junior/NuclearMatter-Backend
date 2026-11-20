"""Original-invariant support, density, isolated tail and evidence-admission probes."""
from reference import *
from scipy.integrate import quad
from scipy.optimize import brentq
import ast,copy
checks=[];diagnostics=[];start=time.monotonic()
def ck(n,e,g):checks.append(dict(name=n,error=float(e),gate=g,passed=bool(math.isfinite(e)and abs(e)<=g)))
def reject(n,fn):
 try:fn();ok=False;msg='accepted'
 except (ValueError,ArithmeticError,RuntimeError)as e:ok=True;msg=type(e).__name__+':'+str(e)
 except Exception as e:ok=False;msg=type(e).__name__+':'+str(e)
 checks.append(dict(name=n,passed=ok,result=msg))
b,meta=S.matched();pilot=json.loads((P/'pilot_results.json').read_text());en=.835
root=brentq(lambda p:ENDPOINT*(math.hypot(MN,p)+p)/MN-en,0,500.,xtol=1e-12)
formula=MN*(en*en-ENDPOINT*ENDPOINT)/(2*en*ENDPOINT)
ck('minimum parent original Doppler invariant',root/formula-1,3e-12)
ref=next(r['value']for r in pilot['rows']if r['kind']=='independent'and r['value']['Enu']==en and r['value']['controls']['parent']==12)
scale=math.sqrt(2*MN*b.T);density=scale**3/(math.pi**2*HC**3)*quad(lambda x:x*x*expit((b.mu_n-math.hypot(MN,scale*x))/b.T),root/scale,np.inf,epsabs=1e-18,epsrel=3e-12)[0]
ck('allowed parent density independent momentum',ref['density_allowed']/density-1,3e-8)
tail=integral(en,12,16,16,12,100.)
for key in ['j','chi']:ck('independent isolated FD tail85to100 '+key,math.expm1(ref[key+'_log']-tail[key+'_log']),3e-6)
for x in [True,False,np.bool_(True),None,'0.8',float('nan'),float('inf'),0.,-1.,1.400001]:reject('incident guard '+repr(x),lambda x=x:S.integrate(x))
for kw in [dict(parent_order=True),dict(phi_order=0),dict(tail=float('nan')),dict(tail=True),dict(budget_seconds=121.),dict(budget_seconds=True)]:reject('control guard '+repr(kw),lambda kw=kw:S.integrate(.835,**kw))
for log,reason in [(-1000.,'underflow'),(1000.,'overflow')]:
 r=S.represented(log);ck('ordinary soft-output representation '+reason,int(not(r['value']is None and r['log']==log and r['representable']is False and r['reason']==reason)),0)
for log in [False,float('nan'),float('inf')]:reject('nonphysical log guard '+repr(log),lambda log=log:S.represented(log))
# Exact literal consumer functions only: never import or execute total.py driver.
source=(ROOT/'hot_beta_opacity_spectrum/total.py').read_text();tree=ast.parse(source);nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in ['contract','validate']];ns=dict(S=S,np=np,math=math,hashlib=hashlib,leggauss=leggauss,EDGES=[0.,.2,.5,.7,S.ENDPOINT,.9,1.05,1.2,1.4],P=ROOT/'hot_beta_opacity_spectrum')
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(ROOT/'hot_beta_opacity_spectrum/total.py'),'exec'),ns)
stored=json.loads((ROOT/'hot_beta_opacity_spectrum/total_fine_0.json').read_text());contract=ns['contract']('fine');row=stored['rows'][0];ns['validate'](contract,row,b,meta);ck('genuine saved-node admission ONLY',0,0)
for label,edit in [('both false empty',lambda r:r['spectrum'].update(j=S.represented(None),chi=S.represented(None))),('false j empty',lambda r:r['spectrum'].update(j=S.represented(None))),('Boolean index',lambda r:r.update(index=False)),('wrong energy',lambda r:r.update(E=r['E']+.01)),('wrong bath',lambda r:r['spectrum']['bath'].update(T=3.)),('wrong source identity',lambda r:r['spectrum'].update(implementation_sha256='0'*64)),('wrong controls',lambda r:r['spectrum']['controls'].update(parent_order=99))]:
 r=copy.deepcopy(row);edit(r);reject('saved-node admission '+label,lambda r=r:ns['validate'](contract,r,b,meta))
# No soft coefficient accuracy claim: only check it cannot masquerade as exact zero.
for soft in [1e-8,1e-100,1e-160]:
 try:
  v=S.integrate(soft,parent_order=6,cone_order=8,electron_order=8,phi_order=6,tail=60.)
  ok=all(v[k]['log']is not None and math.isfinite(v[k]['log'])and v[k]['reason']!='empty_support'for k in ['j','chi']);diagnostics.append(dict(Enu=soft,status='finite logs, accuracy unqualified',j=v['j'],chi=v['chi']))
 except (ArithmeticError,ValueError)as e:ok=True;diagnostics.append(dict(Enu=soft,status='explicit rejection; excluded soft extension',error=type(e).__name__+':'+str(e)))
 ck('soft extension never false exact-zero '+repr(soft),int(not ok),0)
# Root SHA is rechecked after every physical call by source and here again.
ck('original hot results remain exact',int(hashlib.sha256((ROOT/'hot_beta_moments/results.json').read_bytes()).hexdigest()!=S.HOT_RESULTS_SHA),0)
(P/'controls_results.json').write_text(json.dumps(dict(checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],density_oracle=density,pmin_root=root,tail=tail,soft_diagnostics=diagnostics,consumer_literal_AST_sha=hashlib.sha256(source.encode()).hexdigest(),seconds=time.monotonic()-start),indent=2)+'\n');print(len(checks),[c for c in checks if not c['passed']],time.monotonic()-start,diagnostics)
