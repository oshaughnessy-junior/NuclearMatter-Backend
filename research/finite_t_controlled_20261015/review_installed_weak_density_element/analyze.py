"""Independent time/endpoint quadrature review; never executes upstream drivers."""
from pathlib import Path
import math,json,hashlib,ast,sys,importlib.util,time
import numpy as np
from scipy.integrate import simpson,quad_vec
from scipy.optimize import brentq
P=Path(__file__).resolve().parent;R=P.parent;A=R/'installed_weak_density_element';checks=[];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def ck(name,error,gate):checks.append(dict(name=name,error=float(error),gate=gate,passed=bool(abs(error)<=gate)))
z2=json.loads((P/'RK4_2.json').read_text());z4=json.loads((P/'RK4_4.json').read_text());a8=json.loads((A/'trajectory_8.json').read_text());L=json.loads((P/'VALIDATION_LOCK.json').read_text());g=L['gates'];n0=1e-7;end=.002;details={}
for z in[z2,z4]:
 for r in z['records']:
  q=r['state'];idx=round(r['t']/end*8);a=a8['endpoints'][idx]['state'];ck(f"RK4{z['steps']} vsMidpoint8 T t{r['t']}",q['T']-a['T'],g['time_abs_T_MeV']);ck(f"RK4{z['steps']} vsMidpoint8 Y t{r['t']}",q['Ye']-a['Ye'],g['time_abs_Ye']);ck('EOS inversion absolute specificu',q['energy']/q['nB']-r['vector'][0],g['EOS_inverse_absolute_u'])
 for key,inds in [('energy',[2,3]),('number',[4,5])]:
  ours=sum(z['records'][-1]['vector'][i]for i in inds);names=['E_nue','E_antinue']if key=='energy'else['N_nue','N_antinue'];other=sum(a8['conserved'][k]for k in names);ck(f'RK4{z["steps"]} vsMidpoint8 escaping {key}',ours/other-1,g['time_escape_relative'])
a=z2['records'][-1]['state'];b=z4['records'][-1]['state'];ck('RK4 2to4 T',a['T']-b['T'],g['time_abs_T_MeV']);ck('RK4 2to4 Y',a['Ye']-b['Ye'],g['time_abs_Ye'])
x2=z2['records'][-1]['vector'];x4=z4['records'][-1]['vector'];ck('RK4 2to4 escapingE',(x2[2]+x2[3])/(x4[2]+x4[3])-1,g['time_escape_relative'])
# Fresh first-stage sources are actual RK4 endpoint sources, not interpolatedrates.
q0=a8['initial'];states=[q0]+[r['state']for r in z4['records']];sources=[z4['calls'][i]for i in[0,4,8,12,16]];times=np.linspace(0,end,5);Qn=[];Rn=[];work=[];sig=[]
for q,s in zip(states,sources):
 ck('source at correct actualnB',s['nB']/q['nB']-1,2e-12);ck('source at correct actualY',s['Y']-q['Ye'],2e-12);ck('source at correct actualT',s['T']-q['T'],2e-10)
 n=q['nB'];Q=s['Qe']+s['Qa'];rr=s['R'];zeta=q['proton']['chemical_potential']+q['electron']['chemical_potential']-q['neutron']['chemical_potential'];Qn.append(Q/n);Rn.append(rr/n);work.append(q['pressure']*.25/n);sig.append((-Q-zeta*rr)/(n*q['T']))
Eq=float(simpson(Qn,x=times));Nq=float(simpson(Rn,x=times));Wq=float(simpson(work,x=times));Sq=float(simpson(sig,x=times));xf=z4['records'][-1]['vector'];s0=q0['entropy']/q0['nB'];sf=b['entropy']/b['nB'];grossN=xf[4]+xf[5]
ck('independent endpointSimpson escape',Eq/(xf[2]+xf[3])-1,g['independent_endpoint_quadrature_relative']);ck('independent endpointSimpson Ye',(Nq-(b['Ye']-.3))/grossN,g['independent_endpoint_quadrature_relative']);ck('independent endpointSimpson work',Wq/xf[6]-1,g['independent_endpoint_quadrature_relative']);ck('independent endpointSimpson materialentropy',(Sq-(sf-s0))/abs(sf-s0),g['material_entropy_change_scaled'])
# Actual first source checked against the prior accepted binding anchor.
prior=json.loads((R/'installed_free_matter_binding/weak_base.json').read_text())['source']['channels']
for ch in prior:
 for k,v in prior[ch].items():ck('freshinitialsource '+ch+k,sources[0]['channels'][ch][k]/v-1,g['initial_forward_relative'])
# Independent forward EOS at final material state: extract ONLY two purefunctions
# from frozen prior independent review; no module/driver execution or writes.
path=R/'review_installed_free_matter_binding/probe.py';tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in ['lepton_mu','independent']]
sp=importlib.util.spec_from_file_location('weak_review_independent_momentum',R/'review_production_baryon_port/oracle.py');O=importlib.util.module_from_spec(sp);sp.loader.exec_module(O)
from scipy.special import expit
ns=dict(math=math,np=np,quad_vec=quad_vec,brentq=brentq,expit=expit,O=O,MN=939.56542052,MP=938.27208816,ME=.51099895069,HC=197.3269804,C=1/(math.pi**2*197.3269804**3));exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
e=ns['independent'](b['T'],b['nB'],b['Ye'])
for k in ['energy','pressure','entropy','cv','e_n','e_Y']:ck('freshfinal independentmomentum '+k,b[k]/e[k]-1,3e-9)
# Review frozenauthor evidence scope and retained coarse failures.
summary=json.loads((A/'summary.json').read_text());details=dict(final_RK4=b,final_midpoint8=a8['final'],RK4_2_final=a,independent_endpoint_integrals=dict(energy=Eq,netLe=Nq,work=Wq,material_entropy=Sq),material_entropy_change=sf-s0,independent_final_EOS=e,author_summary=summary,oracle_source=sha(path),RK4_energy_escape=xf[2]+xf[3],RK4_energy_work=xf[6])
m=json.loads((A/'MANIFEST.json').read_text());actual={k:sha(A/k)for k in m['files']};assert actual==m['files'];ck('all author source and coarsefail bytes unchanged',0,0)
out=dict(checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],details=details)
(P/'analysis_results.json').write_text(json.dumps(out,indent=2)+'\n');print('PASS',out['passed'],'FAIL',len(out['failed']));print(out['failed'])
