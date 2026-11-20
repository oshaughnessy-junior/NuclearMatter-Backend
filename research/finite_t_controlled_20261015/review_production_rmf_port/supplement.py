"""Signedthermalfield/actualFD, projectedroot and branchfailure controls."""
from pathlib import Path
import sys,json,math,importlib.util,hashlib,time
import numpy as np
P=Path(__file__).resolve().parent;R=P.parent;A=R/'production_rmf_port';sys.dont_write_bytecode=True;sys.path.insert(0,str(P/'package_copy/src'));from nmbackend.reference import finite_t_rmf as M;from nmbackend.reference import _rmf_moments as G
sp=importlib.util.spec_from_file_location('supplement_independent_oracle',P/'oracle.py');O=importlib.util.module_from_spec(sp);sp.loader.exec_module(O);prior=json.loads((P/'candidate_results.json').read_text());assert not prior['failed'];checks=[];details={};start=time.monotonic();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def ck(name,error,gate):checks.append(dict(name=name,error=float(error),gate=gate,passed=bool(abs(error)<=gate)))
for i,s in enumerate(prior['states']):
 z=json.loads((P/('oracle_state_'+str(i)+'.json'if i<2 else 'oracle_cold_boundary.json')).read_text());K=sum(g['rho_t']-g['B']*g['n_t']/g['chi']for g in[z['neutron'],z['proton']]);phit=K/z['D'];ck('signedPhiT independent '+str(i),(s['phi_temperature_derivative']-phit)/max(abs(phit),.01),3e-6)
 frozen=(z['T']/O.HC)*sum(g['s_t']-g['n_t']**2/g['chi']for g in[z['neutron'],z['proton']]);relax=z['T']/O.HC*K*K/z['D'];ck('fieldrelaxationCv '+str(i),(s['cv_fixed_density_composition']-frozen-relax)/s['cv_fixed_density_composition'],3e-6)
# Candidate re-solved FDladder at one heldout, notjustcomparisonofanalyticforms.
s=prior['states'][0];vals=[]
for step in[.0015,.00075]:
 dt=s['temperature']*step;up=M.state_density(s['temperature']+dt,s['baryon_density'],s['proton_fraction']);dn=M.state_density(s['temperature']-dt,s['baryon_density'],s['proton_fraction']);vals.append(dict(step=step,Cv=(up.energy_density-dn.energy_density)/(2*dt),phiT=(up.phi-dn.phi)/(2*dt),s=-(up.free_energy_density-dn.free_energy_density)/(2*dt)))
rich=lambda k:(4*vals[1][k]-vals[0][k])/3
ck('candidate re-solvedCv Richardson',rich('Cv')/s['cv_fixed_density_composition']-1,3e-6);ck('candidate re-solvedsignedPhiT Richardson',(rich('phiT')-s['phi_temperature_derivative'])/max(abs(s['phi_temperature_derivative']),.01),3e-6);ck('candidate re-solvedentropy Richardson',rich('s')/s['entropy_density']-1,3e-6);details['candidateFD']=vals
# Rootprojection symmetry is not a returned physicalzero state.
t=1/O.HC;mass=O.M;zero=G.projected_number_scalar(t,0,mass);ck('projectednu0 exactNETsymmetry',zero[0],0)
nu,scalar=M._number_root(t,.213*(1-.237),mass,M.SolveControls());gas=G.moments(t,nu,mass);ck('actualT1 projectedrootforward',gas.density/(.213*(1-.237))-1,2e-9);ck('actualT1 projectedscalarmatchesfull',(scalar-gas.scalar)/gas.scalar,2e-9);ck('actualT1 positivefullgas',0 if gas.energy>0 and gas.chi>0 and math.isfinite(gas.log_antiparticle_density)else 1,0);details['projection']=dict(zero_projection=zero,nu=nu,scalar=scalar,fullgas=vars(gas),note='nu0thermalpaircomponentsunderflow; analyticnetzeroisrootbracketonly,notwholephysicalEOSzero')
# Syntheticwideintervalcases testbranchguards, notnewphysicalstates.
for label,fun in [('no_upcrossing',lambda ph:1.),('two_upcrossings',lambda ph:(ph-.3*O.M)*(ph-.5*O.M)*(ph-.7*O.M)),('nonfinite_scan',lambda ph:float('nan'))]:
 try:M._crossing(fun,128);ok=False
 except ArithmeticError:ok=True
 ck('branchguard '+label,0 if ok else 1,0)
# Tail/rtol variation actualcandidate warmstate, no extrapolationorfit.
refined=M.state_density(s['temperature'],s['baryon_density'],s['proton_fraction'],controls=M.SolveControls(rtol=4e-12,tail=85))
for k in['energy_density','pressure','entropy_density','cv_fixed_density_composition','dirac_mass','mu_n','mu_p']:ck('actualcandidate tail60to85 rtolrefine '+k,getattr(refined,k)/s[k]-1,2e-9)
ck('actualcandidate tailH',np.linalg.norm(np.array(refined.hessian_n_p)-np.array(s['hessian_n_p']))/np.linalg.norm(s['hessian_n_p']),2e-9)
manifest=json.loads((A/'MANIFEST.json').read_text());ck('alloriginalauthorfilesunchanged',0 if all(sha(A/k)==h for k,h in manifest['files'].items())else 1,0)
out=dict(passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],checks=checks,details=details,seconds=time.monotonic()-start);(P/'supplement_results.json').write_text(json.dumps(out,indent=2)+'\n');print('PASS',out['passed'],'FAIL',len(out['failed']),'seconds',out['seconds'])
