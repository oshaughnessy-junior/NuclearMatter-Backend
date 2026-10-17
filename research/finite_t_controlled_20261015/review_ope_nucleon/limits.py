"""Selected Born low-p and central/contact projection checks, no author drivers."""
import json,math
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from probe import A,C,gate,checks,HERE,basis,matrix,hashes,before
start=len(checks);rows=[]
for p in [.35,.7,1.4]:
 for ch in ['nn','np']:
  total=2*math.pi*quad(lambda c:A.born_dsigma(ch,ch,p,c),-1,1,epsabs=1e-20,epsrel=2e-12)[0]*(.5 if ch=='nn'else 1.)
  total2=2*math.pi*quad(lambda c:A.born_dsigma(ch,ch,2*p,c),-1,1,epsabs=1e-20,epsrel=2e-12)[0]*(.5 if ch=='nn'else 1.)
  # Actual expansion parameter grows as p²/mπ²; no empirical NN claim.
  gate(str((p,ch))+' low-p derivative coupling p4',total2/(16*total)-1,20*(2*p/140.)**2)
  rows.append(dict(p=p,channel=ch,total=total,twicep=total2,p4_ratio=total2/(16*total)))
for Q in [83.6,147.9]:
 average=sum(matrix(Q*np.eye(3)[i])for i in range(3))/3
 central=np.zeros((16,16),complex)
 for f,(s3,t3,s4,t4)in enumerate(basis):
  for i,(s1,t1,s2,t2)in enumerate(basis):
   sd=2*(s3==s2)*(s4==s1)-(s3==s1)*(s4==s2);td=2*(t3==t2)*(t4==t1)-(t3==t1)*(t4==t2)
   central[f,i]=-C/3*sd*td*(1-140.**2/(Q*Q+140.**2))
 gate(str(Q)+' angular central/contact-plus-Yukawa projection',np.max(abs(average-central))/C,1e-14)
gate('pure model f=0 exactzero',np.max(abs(A.antisymmetrized([8,3,7],[2,-4,6],A.Model(f_piNN=0)))),0)
gate('exact zero momentum directvertex',np.max(abs(A.direct([0,0,0]))),0)
gate('author hashes still unchanged after extra limits',int(hashes()!=before),0)
(HERE/'limits_results.json').write_text(json.dumps(dict(checks=checks[start:],passed=sum(x['passed']for x in checks[start:]),failed=[x for x in checks[start:]if not x['passed']],rows=rows,author_after=hashes()),indent=2));print(json.dumps(dict(passed=sum(x['passed']for x in checks[start:]),failed=[x for x in checks[start:]if not x['passed']]),indent=2))
