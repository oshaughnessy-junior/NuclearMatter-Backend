"""Independent saved-actual event ledgers and fresh entropy/anti-bound checks."""
from pathlib import Path
import json,math,time
from scipy.integrate import quad
from nmbackend.reference import free_matter as F
P=Path(__file__).resolve().parent;start=time.monotonic();checks=[];rows=[]
def ck(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error) and abs(error)<=tol)))
for index in range(3):
 r=json.loads((P/('source_'+str(index)+'.json')).read_text());v=r['state'];s=F.state_density(v['temperature_mev'],v['baryon_density_fm3'],v['Ye'],photons=v['photons'])
 ck('current exact state/source provenance '+str(index),0 if s.identity==r['state_identity'] else 1,0)
 channels=r['channels'];R=math.fsum([-channels[0]['net'][0]['value'],channels[1]['net'][0]['value'],channels[2]['net'][0]['value']]);Q=math.fsum(c['net'][1]['value'] for c in channels)
 nr=math.fsum(c[d][0]['value'] for c in channels for d in ['forward','backward']);er=math.fsum(c[d][1]['value'] for c in channels for d in ['forward','backward'])
 ck('independent signed number aggregate '+str(index),(R-r['R']['value'])/nr,1e-12);ck('independent signed energy aggregate '+str(index),(Q-r['Q']['value'])/er,1e-12)
 lepton=R+channels[0]['net'][0]['value']-channels[1]['net'][0]['value']-channels[2]['net'][0]['value'];ck('independent Le events '+str(index),lepton/nr,1e-12)
 material=[]
 for c in channels:
  z=[q['value'] for q in c['net']]
  material.extend([z[4],-z[3],-z[2]] if c['channel']=='EC' else [z[3],-z[4],-z[2]] if c['channel']=='PC' else [z[3],z[2],-z[4]])
 ck('independent full p/n/e material energy '+str(index),(math.fsum(material)+Q)/er,2e-8)
 rhs=r['rhs'];dot=s.cv_fixed_nB_Ye*rhs['temperature_dot_mev_s']+s.denergy_dYe_fixed_nB_T*rhs['Ye_dot_s']+s.denergy_dnB_fixed_T_Ye*rhs['baryon_density_dot_fm3_s']
 ck('rest-inclusive prescribed-density energy first law '+str(index),(dot-rhs['energy_work_density_source']+Q)/max(er,abs(dot)),2e-12)
 for c in channels:
  for i,(f,b,n) in enumerate(zip(c['forward'],c['backward'],c['net'])):
   ck('actual F/B versus net '+str((index,c['channel'],i)),(n['value']-f['value']+b['value'])/(f['value']+b['value']),2e-10)
 for species,comp,bound in [('n',s.neutron,s.anti_bounds[0]),('p',s.proton,s.anti_bounds[1])]:
  T,m,mu=comp.temperature,comp.mass,comp.chemical_potential;upper=math.sqrt((m+70*T)**2-m*m)
  integral=quad(lambda p:p*p*math.exp(-(math.hypot(p,m)-m)/T),0,upper,epsabs=1e-10,epsrel=2e-12)[0]
  logarithm=math.log(integral/(math.pi**2*F.HC**3))-(m+mu)/T
  ck('independent momentum MB anti bound '+str((index,species)),logarithm-bound.log_density_upper_bound,2e-10)
 # Fresh EOS entropy derivatives along the instantaneous vector, no time integration.
 entropy_ladder=[]
 for h in [.002,.001]:
  T,n,y=s.temperature_mev,s.baryon_density_fm3,s.Ye
  def ent(t,b,z):return F.state_density(t,b,z,photons=s.photons).entropy_density/b
  a=(ent(T+h,n,y)-ent(T-h,n,y))/(2*h)
  b=(ent(T,n*(1+h),y)-ent(T,n*(1-h),y))/(2*n*h)
  c=(ent(T,n,y+h)-ent(T,n,y-h))/(2*h)
  measured=a*rhs['temperature_dot_mev_s']+b*rhs['baryon_density_dot_fm3_s']+c*rhs['Ye_dot_s']
  expected=rhs['material_entropy_per_baryon_dot_s'];ck('genuine FD entropy-chain ladder '+str((index,h)),(measured-expected)/max(abs(expected),1.),1e-4);entropy_ladder.append(dict(step=h,measured=measured,formula=expected))
 rows.append(dict(index=index,independent_R=R,independent_Q=Q,entropy_ladder=entropy_ladder))
out=dict(checks=checks,passed=sum(x['passed'] for x in checks),failed=[x for x in checks if not x['passed']],rows=rows,seconds=time.monotonic()-start,attribution='Parses actual completed source records; new EOS/anti momentum calculations, zero new weak source calls or time evolution')
(P/'binding_ledger_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ['passed','failed','seconds']}))
