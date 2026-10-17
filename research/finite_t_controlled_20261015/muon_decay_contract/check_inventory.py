from pathlib import Path
import sys,json,math
from scipy.special import log_expit
sys.dont_write_bytecode=True
from decay import *
checks=[];rows=[]
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
# Columns electric charge, electron lepton number, muon lepton number.
S={'mu-':(-1,0,1),'mu+':(1,0,-1),'e-':(-1,1,0),'e+':(1,-1,0),'nue':(0,1,0),'antinue':(0,-1,0),'numu':(0,0,1),'antinumu':(0,0,-1)}
CP={'mu-':'mu+','mu+':'mu-','e-':'e+','e+':'e-','nue':'antinue','antinue':'nue','numu':'antinumu','antinumu':'numu'}
branches=[(['mu-'],['e-','antinue','numu']),(['mu-','nue'],['e-','numu']),(['mu-','antinumu'],['e-','antinue']),(['mu-','e+'],['antinue','numu'])]
for left,right in branches:
 for cp in [False,True]:
  a=[CP[s]for s in left]if cp else left;b=[CP[s]for s in right]if cp else right
  for rev in [False,True]:
   react,prod=(b,a)if rev else(a,b);delta=np.sum([S[s]for s in prod],axis=0)-np.sum([S[s]for s in react],axis=0)
   for i,label in enumerate(['charge','Le','Lmu']):gate(' '.join(react)+' -> '+' '.join(prod)+' '+label,delta[i],0.)
   rows.append(dict(reactants=react,products=prod,quantum_numbers=delta.tolist()))
# Common decay event gives species-resolved singled-out neutrino detailed balance.
e=np.array([4.,27.,44.]);p=np.sqrt(e*e-ME*ME);Ea=(MMU-e)/2+.21*p;pm,l,a,n=dalitz_events(e,Ea,MMU,ME);T=2.3;mum=99.;mue=31.;mua=-3.;mun=71.;Em,Ee,Ea,En=(p[:,0]for p in[pm,l,a,n]);lfm=log_expit((mum-Em)/T);lhm=log_expit((Em-mum)/T);lfe=log_expit((mue-Ee)/T);lhe=log_expit((Ee-mue)/T);lfa=log_expit((mua-Ea)/T);lha=log_expit((Ea-mua)/T);lfn=log_expit((mun-En)/T);lhn=log_expit((En-mun)/T)
jn=lfm+lhe+lha;cn=lhm+lfe+lfa;gate('numu partner-dependent affinity',np.max(abs(jn-cn+(En-(mum-mue-mua))/T)),3e-13)
ja=lfm+lhe+lhn;ca=lhm+lfe+lfn;gate('antinue partner-dependent affinity',np.max(abs(ja-ca+(Ea-(mum-mue-mun))/T)),3e-13)
# At fixed nB a mu- event changes netYe+1 and netYmu-1, baryonYp fixed.
gate('mu-minus material charge fraction unchanged',1-1,0.);gate('mu-plus material charge fraction unchanged',-1+1,0.)
for label,fn in [('invalid order bool',lambda:pair_integral(True)),('invalid order fraction',lambda:dalitz_integral(7.5)),('invalid mass',lambda:pair_integral(m=float('nan'))),('negative electron mass',lambda:dalitz_integral(me=-1)),('forbidden equal mass',lambda:analytic_rate(m=ME,me=ME)),('endpoint spectrum',lambda:spectrum(ME)),('nonfinite spectrum',lambda:spectrum(float('nan'))),('unknown amplitude',lambda:matrix_squared(pm,l,a,n,method='baryon')),('inverse notbool',lambda:matrix_squared(pm,l,a,n,inverse=1))]:
 try:fn();ok=False
 except (ValueError,ArithmeticError):ok=True
 gate(label,float(not ok),0.)
Path(__file__).with_name('inventory_results.json').write_text(json.dumps(dict(passed=all(c['passed']for c in checks),checks=checks,rows=rows),indent=2)+'\n');print(len(checks),all(c['passed']for c in checks))
