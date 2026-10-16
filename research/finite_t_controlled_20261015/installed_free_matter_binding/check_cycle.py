from pathlib import Path
import sys,json,math,importlib.util,time
P=Path(__file__).resolve().parent;R=P.parent;sys.path.insert(0,str(P));import matter as M
sp=importlib.util.spec_from_file_location('binding_cycle_old_eos_functions',R/'hot_electron_capture_moments/electron_capture.py');EC=importlib.util.module_from_spec(sp);sys.modules[sp.name]=EC;sp.loader.exec_module(EC)
a=json.loads((P/'cycle_32.json').read_text());b=json.loads((P/'cycle_64.json').read_text());checks=[];oldchecks=[]
def ck(name,error,gate):checks.append(dict(name=name,error=float(error),gate=gate,passed=bool(abs(error)<=gate)))
ck('32 to64 Tpath refinement',max(abs(x['T']/y['T']-1)for x,y in zip(a['records'],b['records'][::2])),3e-6)
C=M.LOCK['cycle'];oldinitial=EC.matched(C['T0'],C['n0'],C['Ye'])[1]
s0=sum(oldinitial[k]['entropy_density']for k in ['neutron','proton','electron'])/C['n0']+4*math.pi**2*C['T0']**3/(45*M.HC**3*C['n0'])
for row in b['entropy_roots']:
 T,n=row['T_root'],row['nB'];bath,old=EC.matched(T,n,C['Ye']);eg=math.pi**2*T**4/(15*M.HC**3);ent=sum(old[k]['entropy_density']for k in ['neutron','proton','electron'])+4*eg/(3*T);s=M.state(T,n,C['Ye'],photons=True)
 ck('old EOS constantentropy '+str(row['time']),ent/(n*s0)-1,3e-8)
 for field,new in [('energy_density',s.energy),('pressure',s.pressure),('entropy_density',s.entropy)]:
  photon={'energy_density':eg,'pressure':eg/3,'entropy_density':4*eg/(3*T)}[field];val=sum(old[k][field]for k in ['neutron','proton','electron'])+photon;ck('old EOS '+field+' '+str(row['time']),new/val-1,3e-9)
 oldchecks.append(dict(time=row['time'],n=n,T=T,old_entropy_per_baryon=ent/n,new_entropy_per_baryon=s.entropy_per_baryon))
# Independent tighter/tail100 native package quadrature at the actually compressed state.
row=b['records'][16];s=M.state(row['T'],row['nB'],C['Ye'],photons=True);nn=M.B.state_density(s.T,(1-s.Ye)*s.nB,mass_mev=M.MN,rtol=4e-13,tail=100);pp=M.B.state_density(s.T,s.Ye*s.nB,mass_mev=M.MP,rtol=4e-13,tail=100);ee=M.L.state_density(s.T,s.Ye*s.nB,mass_mev=M.ME,rtol=4e-13,tail=100)
for label,x,y in [('n',s.neutron,nn),('p',s.proton,pp),('e',s.electron,ee)]:
 for f in ['energy_density','pressure','entropy_density','chemical_potential']:
  ck('changedstate EOS tail/tolerance '+label+f,getattr(x,f)/getattr(y,f)-1,3e-9)
summary=dict(status='PASS'if all(x['passed']for x in checks)else'FAIL',checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],old_EOS_points=oldchecks,cycle32=a['checks'],cycle64=b['checks'],T_min=min(r['T']for r in b['records']),T_max=max(r['T']for r in b['records']),work64=b['simpson_work'],gross_absolute_work64=b['total_absolute_work'],quarter_work64=b['quarter_work'],return_delta_u64=b['delta_u'])
(P/'cycle_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary['status'],summary['passed'],summary['failed'])
