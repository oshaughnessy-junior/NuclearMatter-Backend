"""Prescribed homogeneous expansion/compression; no hydrodynamic solve.

Classical shared-mass n,p + ideal e+- + photons, static transparent weak rates.
"""
from __future__ import annotations
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import solve_ivp

HERE=Path(__file__).resolve().parent
THERMO=HERE.parent/'thermodynamics'/'lepton_reference.py'
WEAK=HERE.parent/'weak_rates'/'static_capture.py'
sys.dont_write_bytecode=True
sys.path[:0]=[str(THERMO.parent),str(WEAK.parent)]
import lepton_reference as lepton
import static_capture as weak
A=math.pi**2/(15*lepton.HBARC**3)
MASS=939.


def eos(T,n,Y,mu=None):
    if not .25<T<8. or n<=0 or not 0<Y<1:
        raise ValueError('Outside bounded warm dilute composition domain')
    gas=lepton.state_density(T,Y*n) if mu is None else lepton.state_mu(T,mu)
    Y=gas.net_density/n
    if not 0<Y<1: raise ValueError('Unphysical charge composition')
    wavelength=math.sqrt(2*math.pi*lepton.HBARC**2/(MASS*T))
    eta=max(Y,1-Y)*n*wavelength**3/2
    if eta>=.02: raise ValueError('Classical nucleon domain exceeded')
    mixsum=sum(x*math.log(n*x*wavelength**3/2) for x in (Y,1-Y))
    energy=(1-Y)*weak.Q+1.5*T+(gas.energy_density+A*T**4)/n
    pressure=n*T+gas.pressure+A*T**4/3
    entropy=2.5-mixsum+(gas.entropy_density+4*A*T**3/3)/n
    free=(1-Y)*weak.Q+T*(mixsum-1)+(gas.free_energy_density-A*T**4/3)/n
    e_ncharge=gas.chemical_potential-T*gas.dmu_dtemperature_fixed_net_density
    un=(n*Y*e_ncharge-gas.energy_density-A*T**4)/(n*n)
    uT=1.5+gas.cv_fixed_net_density/n+4*A*T**3/n
    uY=-weak.Q+e_ncharge
    return {'gas':gas,'Y':Y,'u':energy,'P':pressure,'s':entropy,'f':free,
            'un':un,'uT':uT,'uY':uY,'eta':eta}


def density_derivative_checks():
    reports=[]
    for T,n,Y in ((.75,1e-8,.4),(2.,1e-7,.3),(5.,1e-5,.6)):
        base=eos(T,n,Y)
        assert abs(base['u']-T*base['s']-base['f'])/max(1.,abs(base['u']))<1e-11
        ladder=[]
        for fraction in (4e-4,2e-4,1e-4):
            h=fraction*n
            lo,hi=eos(T,n-h,Y),eos(T,n+h,Y)
            fdun=(hi['u']-lo['u'])/(2*h)
            fdP=n*n*(hi['f']-lo['f'])/(2*h)
            unerr=abs(n*(fdun-base['un']))/max(1.,abs(n*base['un']))
            Perr=abs(fdP-base['P'])/max(abs(base['P']),1e-30)
            ladder.append({'fraction':fraction,'scaled_density_energy_derivative_error':unerr,'relative_potential_pressure_error':Perr})
        assert ladder[-1]['scaled_density_energy_derivative_error']<1e-6
        assert ladder[-1]['relative_potential_pressure_error']<1e-6
        # Independent entropy derivatives verify reversible compression term.
        hn,ht=1e-4*n,1e-4*T
        sn=(eos(T,n+hn,Y)['s']-eos(T,n-hn,Y)['s'])/(2*hn)
        sT=(eos(T+ht,n,Y)['s']-eos(T-ht,n,Y)['s'])/(2*ht)
        implied=-sn/sT
        actual=(base['P']/n**2-base['un'])/base['uT']
        adiabatic_error=abs(implied-actual)/max(abs(actual),1e-30)
        assert adiabatic_error<1e-6
        reports.append({'T_MeV':T,'nB_fm-3':n,'Ye':Y,'step_ladder':ladder,
                        'relative_entropy_versus_energy_adiabatic_slope_error':adiabatic_error})
    return reports


class Trajectory:
    def __init__(self,factor,duration=.1,n0=1e-7,weak_on=False,order=96):
        self.factor,self.duration,self.n0=float(factor),float(duration),float(n0)
        self.weak_on,self.order=weak_on,order
        self.rate=math.log(self.factor)/self.duration

    def density(self,t):
        n=self.n0*math.exp(self.rate*t)
        return n,self.rate*n

    def rhs(self,t,y):
        T,mu=y[:2]
        n,ndot=self.density(t)
        gas=lepton.state_mu(T,mu)
        state=eos(T,n,gas.net_density/n,mu)
        Y=state['Y']
        if self.weak_on:
            source=weak.source_terms(weak.State(T,mu,n*(1-Y),n*Y),self.order)
            F=source['dYe_dt_s-1']
            q=source['neutrino_loss_MeV_fm-3_s-1']/n
            nu=source['nu_e_number_fm-3_s-1']/n
            anti=source['antinu_e_number_fm-3_s-1']/n
        else:
            F,q,nu,anti=0.,0.,0.,0.
        work=state['P']*ndot/n**2
        Tdot=((state['P']/n**2-state['un'])*ndot-q-state['uY']*F)/state['uT']
        mudot=(n*F+Y*ndot-gas.dn_dtemperature_fixed_mu*Tdot)/gas.dn_dmu_fixed_temperature
        # Local charge and first-law identities include dilution/compression.
        Ydot=(gas.dn_dtemperature_fixed_mu*Tdot+gas.dn_dmu_fixed_temperature*mudot-Y*ndot)/n
        assert abs(Ydot-F)/max(1.,abs(F))<1e-10
        local=state['uT']*Tdot+state['un']*ndot+state['uY']*F-work+q
        assert abs(local)/max(1.,abs(work),abs(q))<1e-11
        return np.array([Tdot,mudot,nu,anti,q,work])


def run(trajectory,rtol):
    T0,Y0=2.,.3
    initial=eos(T0,trajectory.n0,Y0)
    y0=[T0,initial['gas'].chemical_potential,0.,0.,0.,0.]
    def floor(t,y):return y[0]-.8
    floor.terminal,floor.direction=True,-1
    solution=solve_ivp(trajectory.rhs,(0,trajectory.duration),y0,method='DOP853',
                       rtol=rtol,atol=rtol*1e-3,max_step=trajectory.duration/12,
                       dense_output=True,events=floor)
    assert solution.success and not len(solution.t_events[0])
    energyerr=leptonerr=entropyerr=maxeta=0.
    samples=[]
    for t in np.linspace(0,trajectory.duration,121):
        y=solution.sol(t);n,_=trajectory.density(t)
        gas=lepton.state_mu(y[0],y[1]);s=eos(y[0],n,gas.net_density/n,y[1])
        energy=s['u']-initial['u']+y[4]-y[5]
        lepton_residual=s['Y']-Y0+y[2]-y[3]
        energyerr=max(energyerr,abs(energy)/max(1.,abs(initial['u']),abs(y[4]),abs(y[5])))
        leptonerr=max(leptonerr,abs(lepton_residual)/max(1.,abs(y[2])+abs(y[3])))
        entropyerr=max(entropyerr,abs(s['s']-initial['s'])/max(1.,abs(initial['s'])))
        maxeta=max(maxeta,s['eta'])
        samples.append([float(t),n,float(y[0]),s['Y'],s['u'],s['s'],float(y[2]),float(y[3]),float(y[4]),float(y[5])])
    assert energyerr<1e-6 and leptonerr<1e-6
    if not trajectory.weak_on:assert entropyerr<1e-7
    assert np.all(np.diff(np.array(samples)[:,8])>=-1e-12)
    assert np.all(np.diff(np.array(samples)[:,9])*np.sign(trajectory.rate)>=-1e-12)
    return {'density_factor':trajectory.factor,'duration_s':trajectory.duration,'n0_fm-3':trajectory.n0,
            'T0_MeV':T0,'Ye0':Y0,'weak_on':trajectory.weak_on,'weak_order':trajectory.order,'rtol':rtol,
            'final_T_MeV':samples[-1][2],'final_Ye':samples[-1][3],
            'final_escaped_energy_MeV_per_baryon':float(solution.y[4,-1]),
            'final_mechanical_work_on_matter_MeV_per_baryon':float(solution.y[5,-1]),
            'max_scaled_energy_ledger_error':energyerr,'max_scaled_lepton_ledger_error':leptonerr,
            'max_relative_matter_entropy_change':entropyerr,'max_nucleon_n_lambda3_per_spin':maxeta,
            'rhs_evaluations':solution.nfev,
            'sample_columns':['t_s','nB_fm-3','T_MeV','Ye','u_MeV_per_baryon_common_rest_subtracted','entropy_per_baryon_kB','escaped_nue_per_baryon','escaped_antinue_per_baryon','escaped_energy_MeV_per_baryon','mechanical_work_on_matter_MeV_per_baryon'],
            'samples':samples[::10]},solution


def compare(a,b,label):
    ra,sa=a;rb,sb=b
    times=np.linspace(0,ra['duration_s'],101)
    ya,yb=sa.sol(times),sb.sol(times)
    error=float(np.max(abs(ya-yb)/np.maximum(1.,abs(yb))))
    assert error<1e-5
    return {'kind':label,'weak_on':ra['weak_on'],'density_factor':ra['density_factor'],'max_scaled_shared_time_history_difference':error}


def main():
    paths=(Path(__file__).resolve(),THERMO,WEAK)
    hashes={str(p.relative_to(HERE.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    checks=density_derivative_checks()
    print('Density/free-energy pressure and entropy slope checks PASS',flush=True)
    runs=[];comparisons=[]
    # Weak-on execution is gated by completed weak-off work/entropy tests.
    for weak_on in (False,True):
        for factor in (.5,2.):
            group=[]
            configs=((2e-6,96),(2e-9,96),(2e-9,160)) if weak_on else ((2e-6,96),(2e-9,96))
            for tolerance,order in configs:
                result=run(Trajectory(factor,weak_on=weak_on,order=order),tolerance)
                runs.append(result[0]);group.append(result)
                print(f'Completed weak={weak_on} densityfactor={factor} tolerance={tolerance} order={order}',flush=True)
            comparisons.append(compare(group[0],group[1],'ODE tolerance'))
            if weak_on:comparisons.append(compare(group[2],group[1],'weak quadrature'))
    final_hashes={str(p.relative_to(HERE.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    output={'status':'PASS' if hashes==final_hashes else 'REJECTED_DEPENDENCY_CHANGED',
            'utc':datetime.now(timezone.utc).isoformat(),'command':f'{sys.executable} {Path(__file__).resolve()}',
            'source_sha256':hashes,'end_source_sha256':final_hashes,'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
            'constants':{'common_nucleon_kinetic_mass_MeV':MASS,'Q_MeV':weak.Q,'electron_mass_MeV':lepton.ELECTRON_MASS,'a_gamma':A},
            'density_law':'nB(t)=nB0*exp[ln(factor)*t/duration]; prescribed, not hydrodynamically solved',
            'density_derivative_checks':checks,'runs':runs,'refinement_checks':comparisons,
            'publication_status':'Requires independent review before publication',
            'limitations':['Shared-mass classical/static nucleons, no recoil/blocking/interactions/pairing/nuclei/muons',
                           'No hydro, gravity or radiation transport; prescribed density and transparent neutrino escape',
                           'Local electromagnetic/pair equilibrium and photon comoving retention imposed',
                           'Mechanical-work reservoir accounted explicitly; matter alone need not conserve energy',
                           'Matter entropy conservation tested only with weak processes disabled',
                           'No box thermalization or connected-mode claim; static antineutrino support gap retained']}
    (HERE/('results.json' if hashes==final_hashes else 'rejected_run.json')).write_text(json.dumps(output,indent=2)+'\n')
    assert output['status']=='PASS'
    print(json.dumps({'status':'PASS','runs':len(runs),'refinements':len(comparisons),
                      'max_energy_ledger':max(r['max_scaled_energy_ledger_error'] for r in runs),
                      'max_lepton_ledger':max(r['max_scaled_lepton_ledger_error'] for r in runs),
                      'max_weakoff_entropy':max(r['max_relative_matter_entropy_change'] for r in runs if not r['weak_on'])},indent=2))

if __name__=='__main__':main()
