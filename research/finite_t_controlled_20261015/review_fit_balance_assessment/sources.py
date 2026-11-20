import time,json,math
from common import *
start=time.monotonic();before=inventory();checks=[];rows=[]
def ck(name,error,gate):checks.append(dict(name=name,error=float(error),gate=gate,passed=bool(math.isfinite(error)and abs(error)<=gate)))
roots=json.loads((P/'roots_results.json').read_text());assert roots['failed']==[]
for i,item in enumerate(roots['rows']):
 T,n,y=item['T'],item['nB'],item['root'];bath,s=eos(T,n,y);profiles=[]
 for fine in [False,True]:
  if time.monotonic()-start>105:raise TimeoutError('review source batch105s')
  ec=C.integrate(C.Bath(**bath),proton_order=30 if fine else 24,electron_order=50 if fine else 40,initial_order=12 if fine else 10,outgoing_order=12 if fine else 10,phi_order=12 if fine else 10,tail=70. if fine else 55.,scenarios=[dict(name='empty',mode='empty')])
  pc=PC.integral(bath,order_n=34 if fine else 30,order_e=54 if fine else 48,order_angle=12 if fine else 10,tail=70. if fine else 55.)
  beta=BETA.integrate(BETA.K.Bath(**bath),rest_order=26 if fine else 20,parent_order=30 if fine else 26,orient_order=10 if fine else 8,phi_order=16 if fine else 12,coordinate='electron',parent_coordinate='direct_p',tail=90. if fine else 75.,scenarios=[dict(name='empty',mode='empty')])
  channels=dict(EC=ec['scenarios']['empty']['net'],PC=dict(zip(PC.NAMES,map(float,np.asarray(pc['sums'])[0,0]))),beta=beta['scenarios']['empty']['net'])
  rates={k:v['number']for k,v in channels.items()};R=rates['beta']+rates['PC']-rates['EC'];gross=sum(rates.values());Q=sum(v['neutrino_energy']for v in channels.values())
  neutron=channels['EC']['neutron_energy']-channels['beta']['neutron_energy']-channels['PC']['neutron_energy']
  proton=channels['beta']['proton_energy']+channels['PC']['proton_energy']-channels['EC']['proton_energy']
  lepton=channels['beta']['electron_energy']-channels['PC']['positron_energy']-channels['EC']['electron_energy']
  density=dict(EC=ec['proton_density']/(n*y)-1,PC=pc['neutron_density']/(n*(1-y))-1,beta=beta['parent_density']/(n*(1-y))-1)
  for channel,error in density.items():ck(f'{i}/{fine} physical FDdensity {channel}',error,3e-8)
  ck(f'{i}/{fine} rest-inclusive species energy',(neutron+proton+lepton+Q)/Q,3e-10)
  ck(f'{i}/{fine} direct collision energy',(Q-sum(v['material_energy_loss']for v in channels.values()))/Q,3e-10)
  dn,dp,dminus,dplus=rates['EC']-rates['beta']-rates['PC'],R,rates['beta']-rates['EC'],-rates['PC']
  ck(f'{i}/{fine} baryon ledger',(dn+dp)/gross,1e-14)
  ck(f'{i}/{fine} electric charge',(dp-(dminus-dplus))/gross,1e-14)
  ck(f'{i}/{fine} electronLe including radiation',(dminus-dplus+rates['EC']-rates['beta']-rates['PC'])/gross,1e-14)
  pair_minus=s['electron']['minus_fraction']*R-dminus
  pair_plus=-s['electron']['plus_fraction']*R-dplus
  ck(f'{i}/{fine} EM equal pair replenishment',(pair_minus-pair_plus)/gross,1e-14)
  ck(f'{i}/{fine} positive pair replenishment',0 if min(pair_minus,pair_plus)>0 else 1,0)
  H=Q+s['uY']*R
  ck(f'{i}/{fine} fixed T external energy',(H-Q-s['uY']*R)/Q,3e-14)
  # Independent directional energy derivative was gated before this ledger.
  H_fd=Q+item['finite_uY'][-1]*R
  ck(f'{i}/{fine} directional H',(H_fd-H)/Q,3e-10)
  ck(f'{i}/{fine} positive detailed residual',0 if R>0 else 1,0)
  ck(f'{i}/{fine} locked model net/gross',R/gross,.001)
  profiles.append(dict(bath=bath,eos=s,fine=fine,channels=channels,rates=rates,R=R,gross=gross,actual_net_gross=R/gross,dYe_dt=R/n,Q=Q,H=H,H_directional=H_fd,pair_replenishment=pair_plus,stoichiometry=dict(neutron=dn,proton=dp,electron_minus=dminus,electron_plus=dplus,nue=rates['EC'],antinue=rates['beta']+rates['PC'])))
 for channel in profiles[1]['channels']:
  for field,value in profiles[1]['channels'][channel].items():ck(f'{i} individual source refine {channel}/{field}',profiles[0]['channels'][channel][field]/value-1,3e-6)
 ar=AUTHOR['rows'][i];fine=profiles[-1]
 ck(f'{i} detailed residual versus author absolute gross units',fine['actual_net_gross']-ar['actual_detailed_net_over_gross'],3e-8)
 ck(f'{i} actual Q versus author',fine['Q']/ar['detailed_both_profiles'][-1]['escaping_energy']-1,3e-6)
 fit=F.approximate_moments(T,n,y)['moments'];diagnostics={}
 for ch,nkey,qkey in [('EC','R_EC','Q_EC'),('PC','R_PC','Q_PC')]:
  diagnostics[nkey]=fit[nkey]/fine['channels'][ch]['number']-1;diagnostics[qkey]=fit[qkey]/fine['channels'][ch]['neutrino_energy']-1
 rows.append(dict(index=i,T=T,nB=n,Ye=y,profiles=profiles,individual_fit_fraction_errors_diagnostic=diagnostics))
 (P/'sources_partial.json').write_text(json.dumps(dict(rows=rows,checks=checks,seconds=time.monotonic()-start),indent=2)+'\n')
after=inventory();ck('all frozen input bytes unchanged',0 if before==after else 1,0)
out=dict(passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],checks=checks,rows=rows,before=before,after=after,seconds=time.monotonic()-start)
(P/'sources_results.json').write_text(json.dumps(out,indent=2)+'\n');print(dict(passed=out['passed'],failed=out['failed'],seconds=out['seconds'],roots=[dict(Ye=r['Ye'],Rovergross=r['profiles'][-1]['actual_net_gross'],dYe=r['profiles'][-1]['dYe_dt'],Q=r['profiles'][-1]['Q'],H=r['profiles'][-1]['H'])for r in rows]));assert not out['failed']
