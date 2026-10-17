"""Primal/thermodynamic qualification of independent oracle BEFOREcandidate."""
from pathlib import Path
import importlib.util,sys,json,time,math,hashlib
import numpy as np
P=Path(__file__).resolve().parent;s=importlib.util.spec_from_file_location('independent_pure_rmf_oracle',P/'oracle.py');O=importlib.util.module_from_spec(s);s.loader.exec_module(O);L=json.loads((P/'PREPARATION_LOCK.json').read_text());checks=[];states=[];start=time.monotonic()
def ck(name,error,gate):checks.append(dict(name=name,error=float(error),gate=gate,passed=bool(abs(error)<=gate)))
def save():
 (P/'pure_results.json').write_text(json.dumps(dict(candidate_imported=False,source='independentmomentumonly',passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],checks=checks,states=states,seconds=time.monotonic()-start),indent=2)+'\n')
for i,point in enumerate(L['heldouts_nB_T_Yp']+[L['spinodal_diagnostic_nB_T_Yp']]):
 z=json.loads((P/f'oracle_state_{i}.json').read_text())if i==0 else O.state(*point);(P/f'oracle_state_{i}.json').write_text(json.dumps(z,indent=2)+'\n');states.append(z)
 n,T,Y=point
 ck('pure netroots '+str(i),max(abs(r)for r in z['number_residuals']),2e-9);ck('pure scalar '+str(i),z['scalar_residual']/n,2e-9)
 ck('pure Euler '+str(i),(z['energy']+z['pressure']-T*z['entropy']-n*((1-Y)*z['mu_n']+Y*z['mu_p']))/(T*z['entropy']),2e-9)
 ck('pure grandpotential '+str(i),(sum(v['grandpotential']for v in[z['neutron'],z['proton']])+sum(v['pressure']for v in[z['neutron'],z['proton']]))/sum(v['pressure']for v in[z['neutron'],z['proton']]),2e-9)
 ck('pure scalartrace '+str(i),max(abs(v['scalar']-(v['energy']-3*v['pressure'])/(z['mD']/O.HC))for v in[z['neutron'],z['proton']])/n,2e-9)
 if i<2:ck('pure heldout locallystable '+str(i),0 if min(z['H_eigenvalues'])>0 and z['D']>0 and z['Cv']>0 else 1,0)
 else:
  H=np.array(z['H']);v=np.array([1-Y,Y]);ck('pure spinodalnegativefullH',0 if min(z['H_eigenvalues'])<0 else 1,0);ck('pure spinodalpositiveDandfixedY',0 if z['D']>0 and v@H@v>0 else 1,0)
 save()
for i,(t,nu,m)in enumerate(L['pair_gas_t_nu_m_internal_fm']):
 z=O.gas(t,nu,m);base=[t,nu,m];responses=[('chi','net',1,1),('n_t','net',0,1),('B','scalar',1,1),('B','net',2,-1),('Q','scalar',2,1),('rho_t','scalar',0,1),('s_t','entropy',0,1)]
 for name,field,j,sign in responses:
  estimates=[]
  for rh in L['FD_steps']:
   step=rh*max(abs(base[j]),.2);up=base.copy();dn=base.copy();up[j]+=step;dn[j]-=step
   estimates.append(sign*(O.gas(*up)[field]-O.gas(*dn)[field])/(2*step))
  rich=(4*estimates[-1]-estimates[-2])/3;scale=max(abs(z[name]),.01)
  ck('pure gasFD '+str(i)+name+str(j),(rich-z[name])/scale,2e-6)
 # Finite difference pairenergy along independent fixed NET root, notfixedparticles.
 vals=[]
 for rh in L['FD_steps']:
  h=rh*t
  def en(tt):return O.gas(tt,O.nu_density(tt,z['net'],m),m)['energy']
  vals.append((en(t+h)-en(t-h))/(2*h))
 rich=(4*vals[-1]-vals[-2])/3;ck('pure fixednetpairCv '+str(i),rich/z['pair_Cv_fixed_net']-1,3e-6)
 ck('pure pair-abundant diagnostic '+str(i),0 if z['anti']/z['particle']>.2 else 1,0)
 (P/f'pair_gas_{i}.json').write_text(json.dumps(z,indent=2)+'\n');save()
save();print('PUREPASS',sum(c['passed']for c in checks),'FAIL',sum(not c['passed']for c in checks),'seconds',time.monotonic()-start)
