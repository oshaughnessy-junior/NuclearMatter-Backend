"""Independentreference convergence/ledger/occupancy audit, no candidateruntime."""
from pathlib import Path
import json,math,ast,hashlib,datetime
import numpy as np
P=Path(__file__).resolve().parent;R=P.parent;L=json.loads((P/'PREPARATION_LOCK.json').read_text());checks=[];details=[];beta=json.loads((P/'beta_base_continuation.json').read_text())['rows']
def ck(name,e,g):checks.append(dict(name=name,error=float(e),gate=g,passed=bool(abs(e)<=g)))
def normalized(z,ch):
 if ch=='PC':
  return {name:dict(F=np.array(z['sums'][j][0]),B=np.array(z['sums'][j][1]),net=np.array(z['net'][j]),entropy=z['entropy'][j])for j,name in enumerate(['empty','LTE','emission_dominated','inverse_dominated'])}
 return {name:dict(F=np.array(v['F']),B=np.array(v['B']),net=np.array(v['net']),entropy=v['entropy'])for name,v in z['scenarios'].items()if name!='unblocked'}
for i in[0,1]:
 fine=json.loads((P/f'pure_{i}_fine.json').read_text());coarse=json.loads((P/f'pure_{i}_base.json').read_text());assert fine['stage']=='COMPLETE'and not fine['failed'];coarse['channels']['NEUTRON_DECAY']=beta[i]['channels']['NEUTRON_DECAY'];allchannels={}
 for ch,z in fine['channels'].items():
  v=normalized(z,ch);w=normalized(coarse['channels'][ch],ch);allchannels[ch]=v
  for case,x in v.items():
   F,B,net=x['F'],x['B'],x['net'];gross=F[0]+B[0];EQ=F[1]+B[1];pref=f'bath{i} {ch} {case}'
   for branch in['F','B']:
    for k in range(6):
     target=x[branch][k];value=w[case][branch][k]
     if target>0:ck(pref+' refinement '+branch+str(k),value/target-1,3e-6)
     else:ck(pref+' physicalempty '+branch+str(k),value,0)
   ck(pref+' directnetconsistency',max(abs(net-(F-B))/np.maximum(abs(F)+abs(B),1e-300)),3e-9)
   loss=(net[4]-net[3]-net[2]+net[1])if ch=='EC'else(net[3]-net[4]-net[2]+net[1])if ch=='PC'else(net[3]+net[2]-net[4]+net[1])
   ck(pref+' restinclusivespeciesE',loss/EQ,3e-9);ck(pref+' directmaterialE',(net[5]-net[1])/EQ,3e-9)
   if case=='empty':ck(pref+' inverseexactlyabsent',max(abs(B)),0)
   else:
    ck(pref+' bothpositive',0 if min(F.min(),B.min())>0 else 1,0)
    rad=L['baths'][i]['radiation'][case];bath=L['baths'][i];zeta=bath['mu_p']+bath['mu_e']-bath['mu_n'];eq=zeta if ch=='EC'else -zeta;mu=rad['mu_nue'if ch=='EC'else'mu_antinue'];force=(eq-mu)/bath['T']
    ck(pref+' actualFBratiolog',max(abs(np.log(F/B)-force)),3e-9);ck(pref+' directentropyVsaffinity',(x['entropy']-force*net[0])/gross,3e-9);ck(pref+' entropynegative',min(0.,x['entropy'])/gross,3e-9)
    if case=='LTE':ck(pref+' netgrossLTE',abs(net[0])/gross,2e-10)
    else:ck(pref+' netdirection',0 if net[0]*force>0 else 1,0)
 # CombinedsourceintegersandrestspeciesE, independentlyassembledfromactualnets.
 rows={}
 for case in L['baths'][i]['radiation']:
  ec,pc,be=[allchannels[ch][case]['net']for ch in['EC','PC','NEUTRON_DECAY']];Rnet=be[0]+pc[0]-ec[0];Qnet=be[1]+pc[1]+ec[1];grossR=sum(allchannels[ch][case]['F'][0]+allchannels[ch][case]['B'][0]for ch in allchannels);grossQ=sum(allchannels[ch][case]['F'][1]+allchannels[ch][case]['B'][1]for ch in allchannels)
  neutronN=ec[0]-pc[0]-be[0];protonN=Rnet;electronMinusN=be[0]-ec[0];positronN=-pc[0];LeNu=ec[0];LeAnti=pc[0]+be[0]
  ck(f'bath{i} {case} combinedbaryon',(neutronN+protonN)/grossR,2e-12);ck(f'bath{i} {case} combinedcharge',(protonN-electronMinusN+positronN)/grossR,2e-12);ck(f'bath{i} {case} combinedLe',(electronMinusN-positronN+LeNu-LeAnti)/grossR,2e-12)
  dEn=ec[4]-pc[4]-be[4];dEp=pc[3]+be[3]-ec[3];dEl=be[2]-ec[2]-pc[2];ck(f'bath{i} {case} combinedE',(dEn+dEp+dEl+Qnet)/grossQ,3e-9);rows[case]=dict(Rmatter=Rnet,Qneutrino=Qnet,gross_event=grossR,gross_energy=grossQ,neutron_energy=dEn,proton_energy=dEp,charged_lepton_energy=dEl)
 details.append(dict(bath_index=i,forward=fine['forward'],combined=rows))
# Occupancy ASTproof independentlyreadsselecteddefinitions, no executesource.
def assignments(path,fun,targets):
 tree=ast.parse(path.read_text());f=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==fun)
 return [ast.dump(n,include_attributes=False)for n in ast.walk(f)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id in targets for t in n.targets)]
for ch,old,new,fun,targets in[
 ('EC',R/'review_hot_electron_capture/probe.py',P/'weak_review_ec_phase_trace.py','integrate',{'F','B'}),
 ('PC',R/'review_pc_moments/reference.py',P/'weak_review_pc_phase_polynomial.py','integral',{'F','B','WF','WB'}),
 ('beta',R/'review_crossed_moments/reference.py',P/'weak_review_beta_flatdalitz.py','integrate',{'lf','lb','forward','backward'})]:ck('unchangedactualoccupationproducts '+ch,0 if assignments(old,fun,targets)==assignments(new,fun,targets)else 1,0)
for rel,h in L['sourcepins'].items():ck('sourceunchanged '+rel,0 if hashlib.sha256((R/rel).read_bytes()).hexdigest()==h else 1,0)
coarsefail=[dict(bath=i,failures=json.loads((P/f'pure_{i}_base.json').read_text())['failed'],stage=json.loads((P/f'pure_{i}_base.json').read_text())['stage'])for i in[0,1]]
out=dict(status='PURE fine selectedqualification; originalcoarsePCdensityfailuresretained',passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],checks=checks,details=details,original_coarse_failures=coarsefail,candidate_imported=False,utc=datetime.datetime.now(datetime.timezone.utc).isoformat());(P/'pure_analysis.json').write_text(json.dumps(out,indent=2)+'\n');print('PUREANALYSIS',out['passed'],'NEWFAIL',len(out['failed']));print(out['failed'][:10])
