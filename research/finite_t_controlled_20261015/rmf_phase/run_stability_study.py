"""Bounded local Hessian map, independent derivative ladders and fit holdouts."""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import time
import numpy as np
from stability import *
from gm1_finite_t import sommerfeld_thermal

ROOT=Path(__file__).resolve().parent
start=time.monotonic();checks=[]
def check(name,actual,expected,tol,scale=1e-30):
    error=abs(actual-expected)/max(abs(expected),scale)
    checks.append(dict(name=name,actual=float(actual),expected=float(expected),scaled_error=float(error),tolerance=tol,passed=bool(error<tol)))

selected=[(.02,1.,.1),(.06,1.,.5),(.08,10.,.3),(.12,5.,.1),(.16,10.,.1),(.24,20.,.3),(.32,30.,.1),(.8,20.,.05)]
reviewed=[]
for n,T,Y in selected:
    s=state(n,T,Y);H,detail=implicit_hessian(s)
    label=f'n={n},T={T},Y={Y}'
    fds=[fd_hessian(n,T,Y,h,anchor=s) for h in (2e-3,1e-3,5e-4)]
    rich=(4*fds[2]-fds[1])/3
    rich_lo=(4*fds[1]-fds[0])/3
    scale=max(np.linalg.norm(H,ord=2),1.)
    check(label+' implicit vs chemical FD',np.linalg.norm(H-rich),0.,2e-6,scale)
    check(label+' Richardson ladder',np.linalg.norm(rich-rich_lo),0.,3e-6,scale)
    check(label+' raw chemical Hessian symmetry',np.linalg.norm(fds[2]-fds[2].T),0.,3e-6,scale)
    check(label+' canonical scalar curvature positive',float(detail['canonical_scalar_curvature_fm_minus2']>0),1.,1e-12)
    # Independent free-energy second differences, with deliberately larger steps
    # to avoid subtracting O(rest mass) energies at tiny density increments.
    pair=np.array([n*(1-Y),n*Y]); steps=.003*pair
    def at(nn,np_):return state(nn+np_,T,np_/(nn+np_),initial=s)
    def f(pair_):return at(*pair_).free_energy_density
    f0=s.free_energy_density;HH=np.zeros((2,2))
    for j in range(2):
        dp=np.zeros(2);dp[j]=steps[j]
        HH[j,j]=(f(pair+dp)-2*f0+f(pair-dp))/(steps[j]**2)
    h,k=steps;nn,np_=pair
    HH[0,1]=HH[1,0]=(f(pair+[h,k])-f(pair+[h,-k])-f(pair+[-h,k])+f(pair+[-h,-k]))/(4*h*k)
    check(label+' free-energy Hessian',np.linalg.norm(H-HH),0.,1e-4,scale)
    # Independent fixed-Y pressure and fixed-pressure diffusion slopes.
    hn=.001*n
    pm=state(n-hn,T,Y,initial=s);pp=state(n+hn,T,Y,initial=s)
    result=classify(s,H,detail)
    check(label+' mechanical pressure slope',result['mechanical_dP_dn_fixed_T_Yp'],(pp.pressure-pm.pressure)/(2*hn),2e-4,1.)
    if result['fixed_Y_mechanically_stable']:
        hy=2e-4
        A=np.array([1-Y,Y])@H@np.array([1-Y,Y])
        B=np.array([1-Y,Y])@H@np.array([-1.,1.])
        # Linear compensation of n keeps pressure unchanged to O(hy^2).
        dn_dY=-n*B/A
        plus=state(n+hy*dn_dY,T,Y+hy,initial=s)
        minus=state(n-hy*dn_dY,T,Y-hy,initial=s)
        slope=((plus.mu_p_mev-plus.mu_n_mev)-(minus.mu_p_mev-minus.mu_n_mev))/(2*hy)
        check(label+' fixed-pressure diffusive slope',result['diffusive_dmu_difference_dYp_fixed_pressure'],slope,3e-4,1.)
    result['fd_relative_steps']=[.002,.001,.0005]
    result['richardson_hessian']=rich.tolist();result['free_energy_difference_hessian']=HH.tolist()
    reviewed.append(result)

# A deliberately modest structured sample; these nodes are not a certified domain.
map_results=[]
for T in (1.,10.,20.):
    for Y in (.1,.3,.5):
        anchor=None
        for n in (.02,.04,.06,.08,.12,.16,.24,.32):
            s=state(n,T,Y,initial=anchor);anchor=s
            H,detail=implicit_hessian(s)
            result=classify(s,H,detail)
            check(f'map scalar branch n={n},T={T},Y={Y}',float(detail['canonical_scalar_curvature_fm_minus2']>0),1.,1e-12)
            map_results.append(result)

# Blocked holdouts: all density/proton-fraction nodes differ from the first
# checkpoint's3sampled asymptotic states. No coefficient fitting is performed.
holdouts=[]
for n in (.11,.21,.45,.65):
    for Y in (.07,.23,.41):
        cold=cold_state(n,Y)
        for T in (.05,.2):
            s=state(n,T,Y);fit=sommerfeld_thermal(n,T,Y)
            actual=dict(delta_energy_density=s.energy_density-cold.energy_density,
                        delta_free_energy_density=s.free_energy_density-cold.free_energy_density,
                        delta_pressure=s.pressure-cold.pressure,entropy_density=s.entropy_density)
            actual['thermal_index']=1+actual['delta_pressure']/actual['delta_energy_density']
            errors={key:abs(actual[key]-fit[key])/max(abs(actual[key]),1e-20) for key in actual}
            for key in actual:check(f'heldout n={n},Y={Y},T={T} '+key,fit[key],actual[key],.005)
            hn=2e-4*n
            fp=sommerfeld_thermal(n+hn,T,Y)['delta_free_energy_density']
            fm=sommerfeld_thermal(n-hn,T,Y)['delta_free_energy_density']
            pf=n*(fp-fm)/(2*hn)-fit['delta_free_energy_density']
            check(f'fit pressure derivative consistency n={n},Y={Y},T={T}',fit['delta_pressure'],pf,5e-6)
            holdouts.append(dict(nB=n,T=T,Yp=Y,reference=actual,approximation=fit,relative_errors=errors))

counts={key:sum(x['local_label']==key for x in map_results) for key in ('locally_stable','spinodal','marginal_unqualified')}
counterexamples=[x for x in map_results if x['fixed_Y_mechanically_stable'] and x['local_label']=='spinodal']
report=dict(passed=all(x['passed'] for x in checks),checks=checks,reviewed_states=reviewed,
            map=map_results,map_counts=counts,mechanically_stable_but_spinodal=counterexamples,
            sommerfeld_holdouts=holdouts,holdout_max_relative_error={key:max(x['relative_errors'][key] for x in holdouts) for key in holdouts[0]['relative_errors']},
            runtime_seconds=time.monotonic()-start,
            frozen_reference_sha256=hashlib.sha256((ROOT.parent/'rmf_reference/gm1_finite_t.py').read_bytes()).hexdigest())
(ROOT/'validation_results.json').write_text(json.dumps(report,indent=2)+'\n')
print(f"{sum(x['passed'] for x in checks)}/{len(checks)} checks pass in {report['runtime_seconds']:.2f}s")
print('Map:',counts,'mechanically-stable-but-spinodal:',len(counterexamples))
print('Holdout maxima:',report['holdout_max_relative_error'])
for x in checks:
    if not x['passed']:print('FAIL',x)
raise SystemExit(0 if report['passed'] else 1)
