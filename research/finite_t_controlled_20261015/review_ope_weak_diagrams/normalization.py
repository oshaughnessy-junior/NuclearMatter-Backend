"""Normalize antisymmetric incoming/outgoing labelled states explicitly."""
from pathlib import Path
import json
import numpy as np
from probe import W,point,direct,P,PERM,compare,gate,checks,HERE,snapshot,before
start=len(checks);p,Q=point(117.3,6.7,[-1.7,2.9,.8],[.24,-.61,.75]);om=6.7
for component in['V0','Vx','Vy','Vz','Ax','Ay','Az']:
 b,_=direct(p,om,Q,component);bi,_=direct(p[[1,0,2,3]],om,Q,component);bo,_=direct(p[[0,1,3,2]],om,Q,component);bio,_=direct(p[[1,0,3,2]],om,Q,component)
 # Two factors1/sqrt2, then the four literal Slater contractions.
 slater=(b-bi@P-P@bo+P@bio@P)/2
 actual=W.amplitude(p,om,Q,component)[np.ix_(PERM,PERM)];compare('normalized two-sided Slater '+component,actual,slater)
 # Specified nn->np: four initial spin states SUMMED, never average isospin.
 ii=[i for i,s in enumerate(__import__('probe').BASIS)if(s[1],s[3])==(0,0)];ff=[i for i,s in enumerate(__import__('probe').BASIS)if(s[1],s[3])==(0,1)];own=float(np.sum(abs(slater[np.ix_(ff,ii)])**2));auth=float(np.sum(abs(W.channel(W.amplitude(p,om,Q,component),'nn','np'))**2));gate('complete channel spin SUM '+component,own/auth-1,3e-12)
gate('unchanged authors after normalization',int(snapshot()!=before),0)
out=dict(checks=checks[start:],passed=sum(c['passed']for c in checks[start:]),failed=[c for c in checks[start:]if not c['passed']],author_after=snapshot())
(HERE/'normalization_results.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:out[k]for k in['passed','failed']},indent=2))
