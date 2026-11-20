"""Independent thermal density roots and frozen original-delta recipe witness.

Reuse of an earlier independent method is an author cross-check, not a new review.
"""
import ast,sys,math,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from numpy.polynomial.legendre import leggauss
import probe as P
ROOT=P.ROOT;HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'relativistic_delta_oracle'));import oracle as O
source=(ROOT/'review_gm1_differential/probe.py').read_text();tree=ast.parse(source);function=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='independent')
scope=dict(math=math,np=np,brentq=brentq,leggauss=leggauss,O=O,expit=P.expit);exec(compile(ast.Module(body=[function],type_ignores=[]),'frozen_independent_GM1_delta_recipe','exec'),scope);reference=scope['independent']
checks=[]
def ck(name,e,t):checks.append(dict(name=name,error=abs(float(e)),limit=t,passed=bool(abs(e)<=t)))
raw=json.loads((HERE/'results.json').read_text());rows=[];start=time.monotonic()
for row in raw['rows']:
 if row['density_factor']!=.97 or row['T']not in(.1,1.):continue
 s=P.A.OrderedState(**row['state']);kin=row['kinematics'];nB=row['nB'];cold=raw['cold_states'][0]['cold'];densities=[nB*(1-cold['Yp']),nB*cold['Yp'],nB*cold['Ye'],nB*cold['Ymu']]
 targets=[(s.dirac_mass,s.total_mu_a-s.potential_a,densities[0],False),(s.dirac_mass,s.total_mu_b-s.potential_b,densities[1],False),(P.B.ME,row['electron']['chemical_potential'],densities[2],True),(P.B.MMU,row['muon']['chemical_potential'],densities[3],True)]
 roots=[]
 for mass,mu,n,pairs in targets:
  kf=(3*math.pi**2*n)**(1/3)*P.B.H;Ef=math.hypot(mass,kf)
  independent=brentq(lambda chemical:P.direct_density(s.T,chemical,mass,pairs)-n,Ef-20*s.T,Ef+20*s.T,xtol=8e-12,rtol=2e-14)
  ck('independent_thermal_mu_root',independent-mu,3e-8);roots.append(dict(mass=mass,target=n,independent_mu=independent,EOS_mu=mu,pairs=pairs))
 # Held-out interior angle, separately cold-window classified; main angle remains -1.
 cosine=-.43;Enu=kin['Enu'];El=kin['El'];ml=kin['lepton_mass'];mul=row['electron']['chemical_potential']
 actual=P.A.differential(s,Enu,El,cosine,ml,mul,energy_order=40,phi_order=8,tail=65.)
 refs=[reference(s,Enu,El,cosine,ml,mul,order=o,tail=65.)for o in(24,40)]
 for name in('chi','j'):
  ck('independent_original_delta_'+name,actual[name]['value']/refs[-1][name]-1,3e-7)
  ck('independent_original_delta_refinement_'+name,refs[0][name]/refs[-1][name]-1,3e-7)
 for mode,name in enumerate(('absorption','emission')):
  scale=max(float(np.max(abs(refs[-1]['W'][mode]))),1e-300)
  ck('independent_full_W_'+name,np.max(abs(actual['response'][name]['tensor']-refs[-1]['W'][mode]))/scale,3e-7)
 ck('independent_total_energy_delta',refs[-1]['max_delta'],3e-8)
 coldstate=P.B.state(nB);interior=P.kinematics(coldstate,cosine=cosine);cw=P.cold_window(coldstate,interior,mass=s.dirac_mass,Ua=s.potential_a,Ub=s.potential_b);assert cw['absorption_width']==0
 rows.append(dict(T=s.T,closure=row['closure'],state=row['state'],thermal_density_roots=roots,interior_kinematics=interior,cold_window=cw,actual=actual,independent_runs=refs))
out=dict(checks=checks,passed=sum(c['passed']for c in checks),failed=sum(not c['passed']for c in checks),rows=rows,seconds=time.monotonic()-start,scope='New interior heldout witness using prior frozen independent-delta method; not a fresh independent reviewer')
P.save_json(HERE/'heldout_results.json',out);print(out['passed'],out['failed'],out['seconds'])
if out['failed']:raise SystemExit(1)
