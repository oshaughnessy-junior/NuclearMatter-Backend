"""Independent value recurrence, momentum quadrature and fresh GM1 holdouts."""
from pathlib import Path
import sys,json,time,math,hashlib
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import expit
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;R=H.parent;A=R/'gm1_affinity_fit';sys.path.insert(0,str(A));import potential as F
sys.path.insert(0,str(R/'rmf_reference'));import gm1_finite_t as E
start=time.monotonic();checks=[];rows=[]
def hashes():
 extra=['rmf_reference/gm1_finite_t.py','rmf_phase/stability.py','gm1_potential_fit/potential.py','gm1_potential_fit/coefficients.json','review_gm1_fit/results.json']
 return {str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest()for p in list(A.iterdir())+[R/x for x in extra]if p.is_file()}
initial=hashes();coeff=json.loads((A/'coefficients.json').read_text());fields=coeff['fields'];protocol=json.loads((A/'PROTOCOL.json').read_text());pre=json.loads((A/'PRE_HOLDOUT_FREEZE.json').read_text());oldhold=json.loads((A/'HOLDOUT_PROTOCOL.json').read_text())
points=[(.2549,.127,.3251),(.1847,.143,.1879),(.2327,2.47,.3113),(.2589,.121,.3287)]
(H/'REVIEW_PROTOCOL.json').write_text(json.dumps(dict(fresh_points=points,balance_gate=.01,bulk_gates=protocol['bulk_gates'],source_coefficients=initial['gm1_affinity_fit/coefficients.json'],declared_before_EOS=True,coefficients_edits_forbidden=True),indent=2)+'\n')
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
def save(phase):
 (H/'results.json').write_text(json.dumps(dict(phase=phase,passed=phase=='complete'and all(c['passed']for c in checks),checks=checks,rows=rows,initial_hashes=initial,final_hashes=hashes(),runtime_seconds=time.monotonic()-start),indent=2)+'\n')
for filename,sha in pre['files'].items():gate('preholdout hash '+filename,float(hashlib.sha256((A/filename).read_bytes()).hexdigest()!=sha),0.)
gate('196 scalar coefficients',float(sum(np.asarray(c).size for c in fields.values())!=196),0.)
# Independent degree-six Chebyshev VALUE recurrence, complex safe.
def cheb(x):
 vals=[1+0*x,x]
 for _ in range(2,7):vals.append(2*x*vals[-1]-vals[-2])
 return np.array(vals)
def basis(n,Y):
 x=(n-.22)/.04;z=2*((1-2*Y)**2-(.34**2+.64**2)/2)/(.64**2-.34**2)
 return np.outer(cheb(x),cheb(z))
def a(n,T,Y):return 939+sum(T**p*np.sum(np.array(fields[k])*basis(n,Y))for k,p in [('cold',0),('T2',2),('T4',4),('T6',6)])
def fdensity(nn,np_,T):return (nn+np_)*a(nn+np_,T,np_/(nn+np_))
def own_mu(nn,np_,T):
 h=1e-20;return np.array([np.imag(fdensity(nn+1j*h,np_,T))/h,np.imag(fdensity(nn,np_+1j*h,T))/h])
def own_H(nn,np_,T,h):
 return np.column_stack([(own_mu(nn+h,np_,T)-own_mu(nn-h,np_,T))/(2*h),(own_mu(nn,np_+h,T)-own_mu(nn,np_-h,T))/(2*h)])
def forward(n,T,Y):
 if time.monotonic()-start>108:raise TimeoutError('Independent GM1 review108s work bound')
 s=E.state(n,T,Y);gate('EOS forward root '+str((n,T,Y)),s.max_normalized_equation_residual,2e-9)
 if s.max_normalized_equation_residual>2e-9:raise ArithmeticError('Forward root gate failed')
 return s
def own_gas(T,nu,m,order):
 x,w=leggauss(order);end=max(m,abs(nu))+70*T;pmax=math.sqrt((end-m)*(end+m));cuts=sorted({0.,pmax}|{math.sqrt((v-m)*(v+m))for k in[-30,-12,-4,0,4,12,30]if m<(v:=nu+k*T)<end});values=np.zeros(5)
 for lo,hi in zip(cuts[:-1],cuts[1:]):
  p=lo+(x+1)*(hi-lo)/2;weight=w*(hi-lo)/2;en=np.hypot(p,m);z=(en-nu)/T;zb=(en+nu)/T;f=expit(-z);fb=expit(-zb);az=abs(z);ab=abs(zb);entropy=np.logaddexp(0.,-az)+az*expit(-az)+np.logaddexp(0.,-ab)+ab*expit(-ab)
  moments=np.array([f-fb,m/en*(f+fb),en*(f+fb),p*p/(3*en)*(f+fb),entropy])
  values+=np.sum(moments*(p*p*weight)[None,:],axis=1)/math.pi**2
 return values

for n,T,Y in points:
 gate('fresh excludes author holdouts '+str(n),float(any(np.allclose([n,T,Y],p[1:],rtol=0,atol=1e-14)for p in oldhold['points'])),0.)
 gate('fresh excludes training '+str(n),float((n,Y)in[(r['n'],r['Y'])for r in json.loads((A/'development_results.json').read_text())['training']]),0.)
 s=forward(n,T,Y);pred=F.evaluate(n,T,Y);nn=n*(1-Y);np_=n*Y
 # Independent momentum density/scalar/thermodynamic values before derivatives.
 mass=s.dirac_mass_mev/E.HBARC;mom=[]
 for order in[32,48]:
  z=own_gas(T/E.HBARC,s.kinetic_mu_n_mev/E.HBARC,mass,order)+own_gas(T/E.HBARC,s.kinetic_mu_p_mev/E.HBARC,mass,order);mom.append(z)
  gate('independent momentum density '+str((n,T,order)),z[0]/n-1,3e-10)
 gate('momentum quadrature refinement '+str(n),float(np.max(abs(mom[0]-mom[1]))/np.max(abs(mom[1]))),3e-11)
 phi=s.phi_mev/E.HBARC;M=939/E.HBARC;par=E.GM1;us=phi*phi/(2*par['C_sigma'])+par['b']*M*phi**3/3+par['c']*phi**4/4;uv=par['C_omega']*n*n/2+par['C_rho']*(np_-nn)**2/8
 scalar=phi/par['C_sigma']+par['b']*M*phi*phi+par['c']*phi**3
 gate('independent scalar field '+str(n),(scalar-mom[-1][1])/n,3e-10)
 for label,val,ref in [('pressure',(mom[-1][3]-us+uv)*E.HBARC,s.pressure),('energy',(mom[-1][2]+us+uv)*E.HBARC,s.energy_density),('entropy',mom[-1][4],s.entropy_density)]:gate('independent momentum '+label+' '+str(n),val/ref-1,3e-10)
 ownm=own_mu(nn,np_,T);ownf=a(n,T,Y).real;h=1e-20;an=np.imag(a(n+1j*h,T,Y))/h;at=np.imag(a(n,T+1j*h,Y))/h;att=None
 for key,val in [('mu_n',ownm[0]),('mu_p',ownm[1]),('free_per_baryon',ownf),('pressure',n*n*an),('entropy_density',-n*at)]:gate('independent value derivative '+key+' '+str(n),pred[key]/val-1,3e-12)
 # Genuine independent chemical Hessian: separately resolve actual EOS at
 # neutron/proton density perturbations, no implicit_hessian call.
 Hrefs=[];Hfits=[]
 for fraction in[.0002,.0001]:
  h=n*fraction;cols=[]
  for species in[0,1]:
   states=[]
   for sign in[-1,1]:
    densities=np.array([nn,np_]);densities[species]+=sign*h;total=sum(densities);z=forward(total,T,densities[1]/total);states.append(z)
   cols.append((np.array([states[1].mu_n_mev,states[1].mu_p_mev])-np.array([states[0].mu_n_mev,states[0].mu_p_mev]))/(2*h))
  Hrefs.append(np.column_stack(cols));Hfits.append(own_H(nn,np_,T,h))
 gate('exact H ladder '+str(n),np.linalg.norm(Hrefs[0]-Hrefs[1])/np.linalg.norm(Hrefs[1]),3e-6)
 gate('exact H reciprocity '+str(n),np.linalg.norm(Hrefs[1]-Hrefs[1].T)/np.linalg.norm(Hrefs[1]),3e-6)
 gate('fit H independent value derivative '+str(n),np.linalg.norm(Hfits[1]-np.array(pred['chemical_hessian']))/np.linalg.norm(Hfits[1]),3e-6)
 # Independent Cv from actual ENERGY derivative, not entropy differentiation.
 cvs=[]
 for frac in[.003,.0015]:
  h=T*frac;lo=forward(n,T-h,Y);hi=forward(n,T+h,Y);cvs.append((hi.energy_density-lo.energy_density)/(2*h))
 gate('reference energy Cv ladder '+str(n),cvs[0]/cvs[1]-1,3e-5)
 err=dict(balance=abs(math.expm1((pred['mu_n']-pred['mu_p']-s.mu_n_mev+s.mu_p_mev)/T)),mu_difference=pred['mu_n']-pred['mu_p']-s.mu_n_mev+s.mu_p_mev,free=pred['free_per_baryon']-s.free_energy_density/n,energy=pred['energy_density']/n-s.energy_density/n,pressure=pred['pressure']/s.pressure-1,entropy=pred['entropy_density']/s.entropy_density-1,Cv=pred['cv_fixed_nY']/cvs[-1]-1,Hessian=float(np.linalg.norm(np.array(pred['chemical_hessian'])-Hrefs[-1])/np.linalg.norm(Hrefs[-1])))
 for key,tol in [('balance',.01),('free',.005),('energy',.005),('pressure',.01),('entropy',.01),('Cv',.01),('Hessian',.02)]:gate('fresh physical '+key+' '+str(n),err[key],tol)
 gate('fresh positive heat capacity '+str(n),float(pred['cv_fixed_nY']<=0 or cvs[-1]<=0),0.)
 gate('fresh full two-charge stability '+str(n),float(min(np.linalg.eigvalsh(Hrefs[-1])[0],pred['min_chemical_eigenvalue'])<=0),0.)
 rows.append(dict(n=n,T=T,Y=Y,exact=s.asdict(),prediction=pred,independent_H=Hrefs[-1].tolist(),reference_Cv=cvs,errors=err));save('fresh state checkpoint'+str(n))
# Reconstruct design and Sommerfeld targets independently, without fitting.
training=json.loads((A/'development_results.json').read_text())['training'];V=[];VN=[];VY=[];h=1e-20;target_cold=[];target_Y=[];target_n=[];a2_targets=[]
for r in training:
 n,Y=r['n'],r['Y'];V.append(basis(n,Y).ravel());VN.append(np.imag(basis(n+1j*h,Y)).ravel()/h);VY.append(np.imag(basis(n,Y+1j*h)).ravel()/h)
 cold=r['cold'];pfn=E.HBARC*(3*math.pi**2*n*(1-Y))**(1/3);pfp=E.HBARC*(3*math.pi**2*n*Y)**(1/3);md=cold['dirac_mass_mev'];gamma=(pfn*math.hypot(pfn,md)+pfp*math.hypot(pfp,md))/(3*E.HBARC**3);a2=-gamma/(2*n);a2_targets.append(a2)
 gate('independent Sommerfeld target '+str((n,Y)),r['T2_field']/a2-1,3e-14)
 target_cold.append(cold['free_energy_density']/n-939);target_Y.append(cold['mu_p_mev']-cold['mu_n_mev']);target_n.append(cold['pressure']/n**2)
V=np.array(V);VN=np.array(VN);VY=np.array(VY);w=protocol['cold_derivative_weights'];design=np.vstack([w['free']*V,w['aY']*VY,w['an']*VN]);target=np.r_[w['free']*np.array(target_cold),w['aY']*np.array(target_Y),w['an']*np.array(target_n)];cc=np.array(fields['cold']).ravel();residual=design@cc-target
normal=np.linalg.norm(design.T@residual)/(np.linalg.norm(design,2)**2*np.linalg.norm(cc));gate('cold derivative least-squares normal equation',normal,2e-13)
gate('cold design rank49',float(np.linalg.matrix_rank(design)!=49),0.);gate('thermal design rank49',float(np.linalg.matrix_rank(V)!=49),0.)
gate('independent cold condition number',np.linalg.cond(design)/coeff['condition_number_C0']-1,3e-12);gate('independent thermal condition number',np.linalg.cond(V)/coeff['condition_number_other']-1,3e-12)
# Higher-temperature target anchoring and independent thermal LS checks.
for r in training:
 a0=r['cold']['free_energy_density']/r['n']
 for temp,key in [(2.,'warm2'),(3.,'warm3')]:
  reconstructed=a0+r['T2_field']*temp**2+r['T4_field']*temp**4+r['T6_field']*temp**6
  gate('warm target anchor '+str((r['n'],r['Y'],temp)),reconstructed-r[key]['free_energy_density']/r['n'],3e-12)
for name in ['T2','T4','T6']:
 target=np.array([r[name+'_field']for r in training]);c=np.array(fields[name]).ravel();res=V@c-target
 grad=np.linalg.norm(V.T@res)/(np.linalg.norm(V,2)**2*np.linalg.norm(c))
 gate('thermal normal equation '+name,grad,2e-12)
for n,T,Y in points:
 cold=E.cold_state(n,Y);gate('fresh cold forward gate '+str(n),cold.max_normalized_equation_residual,2e-9)
 pfn=E.HBARC*(3*math.pi**2*n*(1-Y))**(1/3);pfp=E.HBARC*(3*math.pi**2*n*Y)**(1/3);md=cold.dirac_mass_mev
 coefficient=-(pfn*math.hypot(pfn,md)+pfp*math.hypot(pfp,md))/(6*n*E.HBARC**3)
 fit_coefficient=float(np.sum(np.array(fields['T2'])*basis(n,Y)))
 gate('fresh Sommerfeld C2 '+str(n),fit_coefficient/coefficient-1,.01)
 rows.append(dict(fresh_cold_C2=dict(n=n,Y=Y,exact=coefficient,fit=fit_coefficient,relative=fit_coefficient/coefficient-1)))
# Even-isospin potential structural identity, outside-domain partner used only
# as a mathematical basis check, never an extra physical holdout.
for Y in[.193,.277,.325]:gate('delta² structural reflection '+str(Y),a(.237,1.3,Y).real-a(.237,1.3,1-Y).real,3e-12)
rows.append(dict(design_audit=dict(condition_C0=float(np.linalg.cond(design)),condition_thermal=float(np.linalg.cond(V)),normalized_normal_equation_residual=normal,cold_free_max=float(np.max(abs(V@cc-target_cold))),cold_aY_max=float(np.max(abs(VY@cc-target_Y))),cold_an_max=float(np.max(abs(VN@cc-target_n))))))
for args in[(.2,.119,.2),(.2,3.001,.2),(.179,.5,.2),(.261,.5,.2),(.2,.5,.331)]:
 try:F.evaluate(*args);rejected=False
 except ValueError:rejected=True
 gate('domain rejects '+str(args),float(not rejected),0.)
gate('all frozen sources coefficients reports unchanged',float(hashes()!=initial),0.);save('complete');print(json.dumps(dict(passed=all(c['passed']for c in checks),checks=len(checks),failures=[c for c in checks if not c['passed']],runtime=time.monotonic()-start)))
