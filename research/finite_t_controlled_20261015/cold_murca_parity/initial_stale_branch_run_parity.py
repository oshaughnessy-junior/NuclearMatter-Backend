"""Execute exact pinned NMB function AST and original compiled NSCool routine."""
from __future__ import annotations
from pathlib import Path
import ast,hashlib,json,math,subprocess,time
import numpy as np
from scipy.optimize import brentq
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
HERE=Path(__file__).resolve().parent;SRC=HERE/'source_snapshot';HC=197.3269804;ME=.51099895069;MM=105.6583755
def nmb_function():
 ns={'jnp':jnp,'np':np}
 for name,names in [('superfluid.py',{'fexp','below','u_1s0','u_3p2b'}),('core_nu.py',{'_sq','_r_mu_a','_r_mu_b','_r_mu_p_3p2','murca'})]:
  tree=ast.parse((SRC/name).read_bytes());nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name in names]
  assert {x.name for x in nodes}==names
  exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SRC/name),'exec'),ns)
 return ns['murca']
NMB=nmb_function()
def evaluate(s):return float(NMB(*[s[k]for k in ['T','kfe','kfm','kfn','kfp','mstn','mstp','tcn','tcp','i','isf']]))
def row(s,ident):return ' '.join(str(x)for x in [ident]+[s[k]for k in ['i','T','kfe','kfm','kfn','kfp','mstn','mstp','tcn','tcp','isf']])+'\n'
def original(states,label):
 inp=''.join(row(s,j+1)for j,s in enumerate(states));(HERE/(label+'_input.txt')).write_text(inp)
 r=subprocess.run([str(HERE/'cold_mu_reference')],input=inp,text=True,capture_output=True,timeout=30)
 (HERE/(label+'_stdout.txt')).write_text(r.stdout);(HERE/(label+'_stderr.txt')).write_text(r.stderr)
 return r,[float(l.split()[1])for l in r.stdout.splitlines()if len(l.split())==2]
def kf(n):return (3*math.pi**2*n)**(1/3)if n>0 else 0.
def cold_state(nB,Yp):
 np_=nB*Yp
 def nl(mu,m):return max(mu*mu-m*m,0.)**1.5/(3*math.pi**2*HC**3)
 mu=brentq(lambda u:nl(u,ME)+nl(u,MM)-np_,ME,500,xtol=1e-12)
 ne,nmu=nl(mu,ME),nl(mu,MM)
 return dict(T=1e9,kfe=kf(ne),kfm=kf(nmu),kfn=kf(nB*(1-Yp)),kfp=kf(np_),mstn=.78,mstp=.69,tcn=0.,tcp=0.,i=5,isf=5,nB=nB,Yp=Yp,Ye=ne/nB,Ymu=nmu/nB,mu_lepton=mu,charge_residual=(ne+nmu-np_)/np_)
def normal_branches(s):
 ab=(1.76-.63*(1.68/s['kfn'])**2)*.68;t=s['T']/1e9
 qn=8.55e21*s['mstn']**3*s['mstp']*(s['kfe']+s['kfm'])/1.68*ab*t**8
 qp=8.55e21*s['mstn']*s['mstp']**3/1.68*(s['kfe']+3*s['kfp']-s['kfn'])**2/(8*s['kfp'])*ab*t**8
 return qn,qp
def build():
 states=[]
 for nb in [.12,.24,.48]:
  for yp in [.04,.08,.14]:
   base=cold_state(nb,yp)
   for T in [2e8,8e8,2e9]:
    for mn,mp in [(.78,.69),(.92,.81)]:states.append(dict(base,T=T,mstn=mn,mstp=mp,label='normal_neutral_cold_lepton_equilibrium'))
 # Pairing is a recipe-convention check, not a new superfluid calculation.
 base=cold_state(.24,.08)
 for tau in [.1,.3,.7,.99]:
  for i in [4,5,6]:
   for mode in ['neutron','proton','both']:
    states.append(dict(base,T=1e9*tau,tcn=0. if mode=='proton'else 1e9,tcp=0. if mode=='neutron'else 1e9,i=i,label='paired_recipe_'+mode))
 for tcn,tcp in [(1e9,0.),(0.,1e9),(1e9,1e9)]:
  for i in [4,5]:states.append(dict(base,T=1e9,tcn=tcn,tcp=tcp,i=i,label='exact_Tc_unpaired_strict_comparison'))
 for key in ['kfn','kfp']:states.append(dict(base,**{key:0.},label='zero_nucleon_guard_'+key))
 states.append(dict(base,T=0.,label='zero_temperature_unpaired_limit'))
 return states
def run():
 start=time.monotonic();states=build();r,qs=original(states,'matched');assert r.returncode==0 and len(qs)==len(states),r.stderr
 checks=[];rows=[]
 def gate(name,error,tol=3e-12):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(np.isfinite(error)and abs(error)<=tol)))
 for j,(s,ref)in enumerate(zip(states,qs)):
  q=evaluate(s);err=q/ref-1 if ref else q;gate('originalFortran vs actualNMB '+str(j),err)
  rowdata=dict(state=s,fortran=ref,NMB=q,relative_error=err)
  if s['label'].startswith('normal'):
   qn,qp=normal_branches(s);gate('independent normal branch sum '+str(j),ref/(qn+qp)-1);gate('matched cold charge '+str(j),s['charge_residual'],3e-13)
   rowdata['formula_branches']={'neutron':qn,'proton':qp}
  rows.append(rowdata)
 base=cold_state(.24,.08);q=evaluate(base)
 for factor in [.37,1.7]:gate('T^8 '+str(factor),evaluate(dict(base,T=base['T']*factor))/q/factor**8-1)
 qn,qp=normal_branches(base)
 for factor in [.7,1.3]:
  gate('neutron effective mass polynomial '+str(factor),evaluate(dict(base,mstn=base['mstn']*factor))/(factor**3*qn+factor*qp)-1)
  gate('proton effective mass polynomial '+str(factor),evaluate(dict(base,mstp=base['mstp']*factor))/(factor*qn+factor**3*qp)-1)
 # Preserve extrapolation/recipe defects separately; agreement is NOT accuracy.
 excluded=[]
 for name,s in [('below_proton_polygon',dict(base,kfn=2.,kfp=.25,kfe=.25,kfm=0.)),('negative_alpha_low_kfn',dict(base,kfn=.8,kfp=.4,kfe=.4,kfm=0.))]:
  out,vals=original([s],name);assert out.returncode==0
  qn,qp=normal_branches(s);excluded.append(dict(name=name,state=s,fortran=vals[0],NMB=evaluate(s),formula_neutron=qn,formula_proton=qp,polygon_slack=s['kfe']+3*s['kfp']-s['kfn'],alpha=1.76-.63*(1.68/s['kfn'])**2,qualified_physical=False))
 for name,s in [('electron_absent',dict(base,kfe=0.)),('no_leptons',dict(base,kfe=0.,kfm=0.))]:
  out,vals=original([s],name);excluded.append(dict(name=name,state=s,fortran_returncode=out.returncode,fortran_stdout=out.stdout,fortran_stderr_file=name+'_stderr.txt',NMB=evaluate(s),qualified_physical=False))
 # Cold formula leaves the muon proton branch out, unlike neutron branch.
 changed=dict(base,kfm=base['kfm']*.61);qn2,qp2=normal_branches(changed)
 gate('proton branch is independent of kfm in this recipe',qp2/qp-1)
 gate('muon change carried only by neutron branch',(evaluate(changed)-q)/(qn2-qn)-1)
 result=dict(checks=checks,passed=sum(x['passed']for x in checks),failed=[x for x in checks if not x['passed']],states=rows,excluded_recipe_diagnostics=excluded,seconds=time.monotonic()-start,NMB_execution='Exact FunctionDef ASTs from pinned core_nu.py/superfluid.py compiled without changing bodies; realjax64 globals. No manual murca rewrite, spline/fullproduction import or driver side effect.',Fortran_execution='Unmodified extracted originallocalnumurca_nucl with exact copied include files; compiled GNUFortran15.2 O0 checkall trapinvalid,zero,overflow',scope='Source-pinned selected cold-prescription parity only; no warmMU/physicalwidth/moderncorrelation validation')
 (HERE/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k]for k in ['passed','failed','seconds']}))
if __name__=='__main__':run()
