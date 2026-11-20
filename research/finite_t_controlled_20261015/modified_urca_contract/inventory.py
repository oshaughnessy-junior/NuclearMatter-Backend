"""Reaction bookkeeping and labelled-state symmetry factors; NO rate kernel."""
from collections import Counter
from pathlib import Path
import math,json
P=Path(__file__).resolve().parent
rows=[];checks=[]
Q=dict(n=0,p=1,em=-1,ep=1,mum=-1,mup=1,nue=0,anue=0,numu=0,anumu=0)
B={k:int(k in ['n','p'])for k in Q};Le={k:0 for k in Q};Lmu=Le.copy()
Le.update(em=1,ep=-1,nue=1,anue=-1);Lmu.update(mum=1,mup=-1,numu=1,anumu=-1)
for flavor,lm,lp,nu,anti in [('e','em','ep','nue','anue'),('mu','mum','mup','numu','anumu')]:
 for spectator in ['n','p']:
  for name,left,right in [('beta',[spectator,'n'],[spectator,'p',lm,anti]),('lepton_capture',[spectator,'p',lm],[spectator,'n',nu]),('antilepton_capture',[spectator,'n',lp],[spectator,'p',anti]),('proton_crossed',[spectator,'p'],[spectator,'n',lp,nu])]:
   for direction,initial,final in [('forward',left,right),('inverse',right,left)]:
    ci,cf=Counter(initial),Counter(final);sym=1/math.prod(math.factorial(v)for v in list(ci.values())+list(cf.values()));changes={k:cf[k]-ci[k]for k in Q if cf[k]!=ci[k]}
    for label,charge in [('Q',Q),('B',B),('Le',Le),('Lmu',Lmu)]:checks.append(dict(name=str((flavor,spectator,name,direction,label)),passed=sum(changes[k]*charge[k]for k in changes)==0))
    checks.append(dict(name='single identical pair factor '+str((flavor,spectator,name,direction)),passed=sym==.5))
    rows.append(dict(flavor=flavor,spectator=spectator,family=name,direction=direction,initial=initial,final=final,changes=changes,labelled_phase_symmetry_factor=sym,meaning='full spin sum, per-spin FD occupations; do not multiply another spin degeneracy'))
(P/'inventory_results.json').write_text(json.dumps(dict(rows=rows,checks=checks,passed=all(c['passed']for c in checks),process_pairs=16,directed_reactions=32,scope='bookkeeping only; muon current and all medium rates UNIMPLEMENTED'),indent=2)+'\n');print(len(checks),all(c['passed']for c in checks))
