"""Independent 1D antinue marginal and thermal vacuum anchors; source pins."""
from pathlib import Path
import json, math, hashlib, sys
from scipy.integrate import quad
from scipy.special import expit
from probe import author,M,me,GF,hbar,hc,H,R
checks=[];data={}
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and abs(error)<=tol)))
manifest=json.loads((R/'muon_thermal_events/MANIFEST.json').read_text())
for group,root in [('files',R/'muon_thermal_events'),('upstream',R)]:
 for name,digest in manifest[group].items():
  gate('frozen '+name,int(hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest),0)
T=3.; mm=author.eos.matter_state(T,8e-5,.25,.015)
b=author.Bath(T,mm.muon.chemical_potential,mm.electron.chemical_potential)
amax=(M*M-me*me)/(2*M)
def marginal(a):
 t=M*M-2*M*a
 return GF*GF*a*a*(t-me*me)**2/(2*math.pi**3*t*hbar)
def mid(a):
 t=M*M-2*M*a
 return (M-a)*(t+me*me)/(2*t)
rest=[quad(lambda a:marginal(a)*f(a),0,amax,epsabs=1e-5,epsrel=2e-12)[0]for f in [lambda a:1,mid,lambda a:a,lambda a:M-a-mid(a)]]
r=(me/M)**2
analytic=GF*GF*M**5/(192*math.pi**3*hbar)*(1-8*r+8*r**3-r**4-12*r*r*math.log(r))
gate('analytic finite mass absolute rest lifetime',rest[0]/analytic-1,3e-12)
gate('rest flavor energy sum',(sum(rest[1:])-M*rest[0])/(M*rest[0]),2e-13)
old=author.vacuum.pair_integral(40)
for val,key in zip(rest,['rate','energy_e','energy_anti_e','energy_nu_mu']):gate('independent rest marginal '+key,val/old[key]-1,3e-11)
den=lambda mu,m:quad(lambda p:p*p/(math.pi**2*hc**3)*expit((mu-math.hypot(m,p))/T),0,700,epsabs=1e-20,epsrel=1e-11)[0]
n=den(b.mu_mu,M)
proper=quad(lambda p:p*p/(math.pi**2*hc**3)*M/math.hypot(M,p)*expit((b.mu_mu-math.hypot(M,p))/T),0,700,epsabs=1e-20,epsrel=1e-11)[0]
ne=den(b.mu_e,me)-den(-b.mu_e,me);nmu=n-den(-b.mu_mu,M)
gate('independent net electron density',ne/(8e-5*.25)-1,3e-10)
gate('independent neutral charge composition',(ne+nmu)/(8e-5*(.25+.015))-1,3e-10)
un=author.integrate(b,None,electron_blocking=False,parent_order=20,pair_order=28,pair_angle=12,orientation_order=8,azimuth_order=12,tail=60.)
gate('fresh moving proper lifetime',un['forward']['number']['value']/(proper*rest[0])-1,3e-10)
for key,val in zip(['electron_energy','antinue_energy','numu_energy'],rest[1:]):gate('fresh exact Doppler '+key,un['forward'][key]['value']/(n*val)-1,3e-10)
gate('fresh parent energy',un['forward']['parent_energy']['value']/(n*M*rest[0])-1,3e-10)
for name,fn in [('mixed temperature',lambda:author.integrate(b,author.Radiation(4,1,2))),('ambiguous blocking',lambda:author.integrate(b,electron_blocking=1)),('boolean quadrature',lambda:author.integrate(b,parent_order=True))]:
 try:fn();rejected=False
 except ValueError:rejected=True
 gate('reject '+name,int(not rejected),0)
# High-inverse-chemical-potential tail *algebra*, not a rate convergence claim.
high=author.Radiation(T,90.,80.)
gate('inverse tail chemical center',author.parent_kinetic_upper(b,high,True,60)-(max(0,b.mu_mu-M,b.mu_e+170-M)+60*T),0)
data.update(rest_independent_marginal=rest,rest_analytic_rate=analytic,mu_minus_density=n,mean_m_over_E=proper/n,net_electron_density=ne,net_muon_density=nmu,fresh_unblocked=un)
(H/'anchors.json').write_text(json.dumps(dict(checks=checks,all_pass=all(c['passed']for c in checks),data=data),indent=2)+'\n')
print(json.dumps(dict(checks=len(checks),failures=[c for c in checks if not c['passed']])))
