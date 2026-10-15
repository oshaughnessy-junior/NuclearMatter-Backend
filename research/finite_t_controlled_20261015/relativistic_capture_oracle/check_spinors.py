from pathlib import Path
import sys,json,math
sys.dont_write_bytecode=True
from events import *
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'muon_full_current'));import current as C
checks=[];rows=[]
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
# External frozen vacuum invariant/source expression supplies a different contraction.
for anti in [False,True]:
 ch=C.V.Channel(C.V.MP if anti else C.V.MN,C.V.MN if anti else C.V.MP,C.V.MMU,anti)
 b=Bath(2,ch.initial_mass,ch.final_mass,940,938,80,150,C.V.MMU,anti)
 for E,c in [(141.3,-.37),(208.1,.61)]:
  p=C.V.momenta(ch,E,c);k,pa,l,pb=(np.array([p[x]])for x in ['k','pa','l','pb']);v=float(amplitude(b,k,pa,l,pb)[0]);q=p['k']-p['l'];t=q[0]**2-q[1:]@q[1:];ff=C.FormFactors().values(t,b.Mweak);ref=sum(C.guo_invariants(ch,p,ff).values());gate(f'{anti} E{E} own spinor vs source invariants',v/ref-1,2e-12)
  beta=np.array([[.12,-.04,.17]]);boosted=[boost(x,beta)for x in [k,pa,l,pb]];vb=float(amplitude(b,*boosted)[0]);gate(f'{anti} E{E} Lorentz invariance',vb/v-1,3e-12)
  for name,x,xx in zip(['k','pa','l','pb'],[k,pa,l,pb],boosted):gate(f'{anti} E{E} {name} boost inversion',np.max(abs(boost(xx,-beta)-x))/max(1,np.max(abs(x))),3e-14)
  rows.append(dict(anti=anti,E=E,cosine=c,spinor_amplitude=v,source_invariant=ref,boosted_amplitude=vb))
Path(__file__).with_name('spinor_results.json').write_text(json.dumps(dict(passed=all(x['passed']for x in checks),checks=checks,rows=rows),indent=2)+'\n');print(len(checks),all(x['passed']for x in checks))
