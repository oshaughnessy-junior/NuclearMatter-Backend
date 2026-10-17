"""Independent isentropic root path and P d(1/n) work quadrature.

Same EOS, different formulation from the temperature/chemical-potential ODE.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import simpson
from scipy.optimize import brentq

sys.dont_write_bytecode=True
from element import eos

HERE=Path(__file__).resolve().parent
path=HERE/'results.json'
measurements=json.loads(path.read_text())
assert measurements['status']=='PASS'
checks=[]
for run in measurements['runs']:
    if run['weak_on'] or run['rtol']!=2e-9:continue
    T0,n0,Y=run['T0_MeV'],run['n0_fm-3'],run['Ye0']
    start=eos(T0,n0,Y)
    factor=run['density_factor']
    cache={}
    def root_state(logratio):
        if logratio not in cache:
            n=n0*math.exp(logratio)
            T=brentq(lambda x:eos(x,n,Y)['s']-start['s'],.5*T0,1.5*T0,xtol=2e-12,rtol=2e-12)
            cache[logratio]=(n,T,eos(T,n,Y))
        return cache[logratio]
    integrals=[]
    for points in (9,17):
        coordinate=np.linspace(0.,math.log(factor),points)
        integrand=[]
        for x in coordinate:
            n,T,state=root_state(float(x))
            integrand.append(state['P']/n)
        integrals.append(float(simpson(integrand,x=coordinate)))
    n,T,finish=root_state(math.log(factor))
    work=finish['u']-start['u']
    relative_work_error=abs(integrals[-1]-work)/max(abs(work),1.)
    relative_quadrature_change=abs(integrals[0]-integrals[1])/max(abs(work),1.)
    relative_ode_temperature_error=abs(T/run['final_T_MeV']-1)
    relative_ode_work_error=abs(run['final_mechanical_work_on_matter_MeV_per_baryon']-work)/max(1.,abs(work))
    assert relative_work_error<2e-6 and relative_quadrature_change<2e-6
    assert relative_ode_temperature_error<1e-7 and relative_ode_work_error<1e-7
    checks.append({'density_factor':factor,'entropy_root_final_T_MeV':T,
                   'EOS_energy_difference_MeV_per_baryon':work,
                   'independent_isentropic_path_work_MeV_per_baryon':integrals,
                   'scaled_work_quadrature_vs_EOS_difference':relative_work_error,
                   'scaled_quadrature_refinement_change':relative_quadrature_change,
                   'relative_ODE_vs_entropy_root_temperature_error':relative_ode_temperature_error,
                   'scaled_ODE_work_vs_EOS_difference_error':relative_ode_work_error})
output={'status':'PASS','command':f'{sys.executable} {Path(__file__).resolve()}',
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'results_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'element_sha256':hashlib.sha256((HERE/'element.py').read_bytes()).hexdigest(),
        'scope':'Same EOS, independent constant-entropy root path and work quadrature; no independent interaction theory',
        'checks':checks}
(HERE/'entropy_oracle_results.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))
