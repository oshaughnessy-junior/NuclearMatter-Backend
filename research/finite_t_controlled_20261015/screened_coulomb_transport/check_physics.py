from pathlib import Path
import sys,json,math
from scipy.integrate import quad
sys.dont_write_bytecode=True
from lorentz import *
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'relativistic_capture_oracle'));from events import spinors
checks=[];rows=[]
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
# Independent Dirac external-potential Born amplitude: average initial electron spin only.
for p in [.011,.31,8.7]:
 E=math.hypot(ME,p);ks2=.017;Z=2;ui=spinors(np.array([[E,0,0,p]]),ME)[0]
 for c in [-.79,.13,.97]:
  uf=spinors(np.array([[E,p*math.sqrt(1-c*c),0,p*c]]),ME)[0];J=uf.conj().T@ui;spin=.5*np.sum(abs(J)**2)/(4*E*E);q2=2*p*p*(1-c);potential=4*math.pi*Z*ALPHA/(q2+ks2);ref=E*E/(4*math.pi**2)*potential*potential*spin
  gate(f'explicit Dirac-Mott prefactor p{p} c{c}',dcs(p,c,ks2,Z)/ref-1,3e-13)
# Direct gain-minus-loss angular operator on an oriented l1 mode.
s=state(.07,1e-7,2);p=HC*(3*math.pi**2*s.ne_fm3)**(1/3);v=p/math.hypot(ME,p);incoming=.37;phi=2*math.pi*np.arange(12)/12
operator=quad(lambda c:s.ni_natural*v*dcs(p,c,s.screening_squared,s.Z)*(2*math.pi/len(phi))*sum(incoming*c-math.sqrt(1-incoming**2)*math.sqrt(1-c*c)*math.cos(f)-incoming for f in phi),-1,1,epsabs=1e-15,epsrel=2e-11)[0]
gate('direct angular operator l1 eigenvalue',operator/(-frequency(p,s)*incoming)-1,3e-11)
f=.83;di=.0021;do=-.0037;gain=(f+do)*(1-f-di);loss=(f+di)*(1-f-do);gate('elastic FD final-blocking cancellation',(gain-loss)/(do-di)-1,3e-12)
# Neutral family Z scaling: ni=ne/Z, so nu is linear in Z at fixedne,ks.
r1=moments(state(.052,2.4e-7,1));r3=moments(state(.052,2.4e-7,3))
for key in ['sigma','kappa_open']:gate('neutral Z scaling '+key,r1[key]/r3[key]/3-1,3e-12)
gate('Seebeck independent of uniform collision scale',r1['seebeck']/r3['seebeck']-1,3e-12)
gate('screening independent of Z atfixedne',r1['state']['screening_squared']/r3['state']['screening_squared']-1,0.)
K=np.array(r1['K']);T=r1['state']['T'];
for X in [(1.,0.),(0.,1.),(.41,-.73)]:
 jn=T*(K[0]*X[0]+K[1]*X[1]);jq=T*(K[1]*X[0]+K[2]*X[1]);entropy=X[0]*jn+X[1]*jq;gate('Onsager entropy quadratic '+str(X),float(entropy<=0),0.)
rows.append(dict(new_heldout_ne=2.4e-7,new_heldout_T=.052,Z1=r1,Z3=r3,angular_operator=operator,eigenvalue_prediction=-frequency(p,s)*incoming))
# Explicit input/domain exclusions, including populated positron baths.
for label,fn in [('pair-rich electron-only excluded',lambda:state(.2,1e-14,1)),('zero electron density',lambda:state(.01,0,1)),('NaN density',lambda:state(.01,float('nan'),1)),('negative temperature',lambda:state(-1,1e-8,1)),('bool charge',lambda:state(.01,1e-8,True)),('fractional ion Z',lambda:state(.01,1e-8,1.2)),('zero screening',lambda:dcs(.1,.2,0,1)),('zero momentum',lambda:frequency(0,s)),('bad cosine',lambda:dcs(.1,2,.1,1)),('NaN screening',lambda:dcs(.1,.1,float('nan'),1)),('tail NaN',lambda:moments(s,tail=float('nan'))),('rtol invalid',lambda:moments(s,rtol=.1)),('unrepresented Coulomb integral',lambda:coulomb_integral(1e300,.3))]:
 try:fn();reject=False
 except (ValueError,ArithmeticError):reject=True
 gate(label,float(not reject),0.)
Path(__file__).with_name('physics_results.json').write_text(json.dumps(dict(passed=all(c['passed']for c in checks),checks=checks,rows=rows),indent=2)+'\n');print(len(checks),all(c['passed']for c in checks))
