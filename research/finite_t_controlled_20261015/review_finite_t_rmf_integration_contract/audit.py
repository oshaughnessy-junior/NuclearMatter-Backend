"""Static provenance and algebra only: no EOS/reference module imported/executed."""
from pathlib import Path
import json,hashlib,ast,re,math,datetime
import numpy as np
P=Path(__file__).resolve().parent;R=P.parent;A=R/'finite_t_rmf_integration_contract';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();checks=[];details={}
def ck(name,error,gate):checks.append(dict(name=name,error=float(error),gate=gate,passed=bool(abs(error)<=gate)))
assert sha(A/'CONTRACT.md')=='f225acd8434612b9d26df3e4e92c167f8c67fcaa0125c9ad485c719bd421d07f';assert sha(A/'MANIFEST.json')=='2f07a52db20d979ca42f71534f40d011e69b421e240cce21be95269292a23803'
m=json.loads((A/'MANIFEST.json').read_text());before={p:sha(A/p)for p in m['files']}
for p,h in m['files'].items():ck('frozen author '+p,0 if before[p]==h else 1,0)
pins=json.loads((A/'SOURCE_PIN.json').read_text())['files']
for p,h in pins.items():ck('pinned source '+p,0 if sha(R/p)==h else 1,0)
HC=197.3269804;Cs=11.785;Co=7.148;Cr=4.410;M=939/HC;b=.002947;c=-.001070
saved=json.loads((A/'SAVED_FORWARD_AUDIT.json').read_text());details['saved_rows']=[]
for i,row in enumerate(saved['rows']):
 src=json.loads((R/row['source']).read_text());loc=row['json_path'];node=src
 for key in re.findall(r'[A-Za-z_]+|\d+',loc):node=node[int(key)]if key.isdigit()else node[key]
 ck('exact saved payload '+str(i),0 if node==row['state']else 1,0);ck('saved external execution flag '+str(i),0 if row['external_code_executed']is False else 1,0)
 z=row['state'];n=z['nB'];Y=z['Yp'];T=z['temperature'];nn=n*(1-Y);pp=n*Y;phi=z['phi_mev']/HC;W=Co*n;rho=Cr*(pp-nn)/2;V=.5*Co*n*n+Cr*(pp-nn)**2/8;U=phi**2/(2*Cs)+b*M*phi**3/3+c*phi**4/4
 ck('saved common939 Dirac mass '+str(i),(z['dirac_mass_mev']+z['phi_mev'])/939-1,2e-12)
 ck('saved omega field '+str(i),z['omega_mev']/(W*HC)-1,2e-12);ck('saved rho halves '+str(i),z['rho_mev']/(rho*HC)-1,2e-12)
 ck('saved vector energy '+str(i),z['vector_field_energy_density']/(V*HC)-1,2e-12);ck('saved scalar nonlinear bareM '+str(i),z['scalar_field_energy_density']/(U*HC)-1,2e-12)
 for sp,sign in [('n',-1),('p',1)]:ck('saved kinetic-total shift '+sp+str(i),(z['mu_'+sp+'_mev']-z['kinetic_mu_'+sp+'_mev']-HC*(W+sign*rho/2))/T,2e-12)
 vecsum=nn*HC*(W-rho/2)+pp*HC*(W+rho/2);ck('vector doublesum identity '+str(i),vecsum/(2*V*HC)-1,2e-12)
 ck('saved f=e−Ts '+str(i),(z['free_energy_density']-z['energy_density']+T*z['entropy_density'])/(T*z['entropy_density']),2e-11)
 ck('saved Euler '+str(i),(z['energy_density']+z['pressure']-T*z['entropy_density']-nn*z['mu_n_mev']-pp*z['mu_p_mev'])/(T*z['entropy_density']),2e-11)
 ck('saved independent energycomponents '+str(i),(z['energy_density']-z['kinetic_energy_density']-z['scalar_field_energy_density']-z['vector_field_energy_density'])/z['energy_density'],2e-12)
 details['saved_rows'].append(dict(n=n,T=T,Y=Y,no_new_EOS=True))
# AST-only cold API/constants, no JAX import, solver or table generation.
source=R/'production_baryon_port/test_mirror/src/nmbackend/eos/rmft.py';tree=ast.parse(source.read_text());gm=None
for node in tree.body:
 if isinstance(node,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='GM1'for t in node.targets):gm={kw.arg:ast.literal_eval(kw.value)for kw in node.value.keywords}
for k,v in dict(M_mev=939.,C_sigma=Cs,C_omega=Co,C_rho=Cr,b=b,c=c).items():ck('literal cold parameter '+k,gm[k]/v-1,0)
cl=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='RMFTEOS');methods={n.name for n in cl.body if isinstance(n,ast.FunctionDef)}
for name in ['symmetric','composition','eos_table','landau_masses']:ck('actual cold class API '+name,0 if name in methods else 1,0)
ck('cold originalsnapshot byte equality',0 if sha(source)==sha(R/'rmf_reference/source_snapshot/rmft.py')else 1,0)
# Independent local thermodynamic-jet oracle. This is exact quadratic algebra,
# NOT a gas EOS or new RMF solution. Omega is a generic consistent grandpotential.
# n=-Omega_nu, rho=Omega_m, s=-Omega_T; all Maxwell identities automatic.
# Eliminate three constraints by direct dense solve, then compare contractSchur.
rng=np.random.default_rng(429381);jetrows=[]
for j in range(12):
 chi=rng.uniform(.3,1.7,2);B=rng.uniform(-.5,.8,2);Q=rng.uniform(-.1,.4,2);nt=rng.uniform(-.7,.7,2);rt=rng.uniform(-.5,.6,2);st=nt*nt/chi+rng.uniform(.3,1.4,2);upp=1.3+abs(sum(Q));D=upp+sum(Q+B*B/chi)
 J=np.array([[upp+sum(Q),-B[0],-B[1]],[B[0],chi[0],0],[B[1],0,chi[1]]]);rhsT=np.array([sum(rt),-nt[0],-nt[1]]);zT=np.linalg.solve(J,rhsT);K=sum(rt-B*nt/chi);temp=1.7
 ck('jet thermalPhi '+str(j),zT[0]-K/D,2e-14)
 # sT along constrainedtrajectory = sum(st+nt*nuT+rt*PhiT).
 cv_direct=temp*sum(st+nt*zT[1:]+rt*zT[0]);cv_formula=temp*(sum(st-nt*nt/chi)+K*K/D);ck('jet relaxed Cv '+str(j),cv_direct/cv_formula-1,2e-14)
 V=np.array([[Co+Cr/4,Co-Cr/4],[Co-Cr/4,Co+Cr/4]]);hh=np.zeros((2,2))
 for k in range(2):
  rhs=np.zeros(3);rhs[k+1]=1;dz=np.linalg.solve(J,rhs);hh[:,k]=dz[1:]+V[:,k]
 aa=B/chi;formula=np.diag(1/chi)-np.outer(aa,aa)/D+V;ck('jet full H '+str(j),np.max(abs(hh-formula)),2e-14);ck('jet symmetric H '+str(j),np.max(abs(hh-hh.T)),2e-14)
 # Explicit vectorpotential partials; fixes rho quarter convention.
 nn=.1+.003*j;pp=.04+.002*j;vp=lambda a,d:Co*(a+d)**2/2+Cr*(d-a)**2/8;h=2e-4
 fd=(vp(nn+h,pp+h)-vp(nn+h,pp-h)-vp(nn-h,pp+h)+vp(nn-h,pp-h))/(4*h*h)
 ck('jet vector mixed partial '+str(j),fd/V[0,1]-1,2e-10)
 jetrows.append(dict(D=D,PhiT_direct=zT[0],PhiT_formula=K/D,Cv_direct=cv_direct,Cv_formula=cv_formula,H_direct=hh.tolist(),H_formula=formula.tolist()))
details['algebra_jets']=jetrows;details['algebra_scope']='Exactconsistentlocalquadraticgrandpotential elimination only, not physicalgas/RMF/EOSexecution.'
phase=json.loads((R/'rmf_phase/validation_results.json').read_text());candidates=phase['mechanically_stable_but_spinodal'];details['saved_spinodal_counterexamples']=candidates
match=next(q for q in candidates if abs(q['nB']-.06)<1e-12 and abs(q['T']-1)<1e-12 and abs(q['Yp']-.1)<1e-12)
ck('saved counterexample positive fixedY',0 if match['mechanical_dP_dn_fixed_T_Yp']>0 else 1,0);ck('saved counterexample fullH negative',0 if min(match['eigenvalues_mev_fm3'])<0 else 1,0)
ck('saved counterexample canonicalDpositive',0 if match['implicit_response']['canonical_scalar_curvature_fm_minus2']>0 else 1,0)
assert {p:sha(A/p)for p in m['files']}==before;assert all(sha(R/p)==h for p,h in pins.items());ck('all45sourcepins unchanged afteraudit',0,0)
out=dict(passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],checks=checks,details=details,sourcepins=pins,authorfiles=before,no_new_physics_or_solver_executed=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat());(P/'results.json').write_text(json.dumps(out,indent=2)+'\n');print('PASS',out['passed'],'FAIL',len(out['failed']))
