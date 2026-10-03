"""Fixed-mass conditional inner-core sensitivities; no production inversion API."""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq
from nmbackend.eos import MetaModelEOS
from nmbackend.eos.coolcomp import cooling_composition
from nmbackend.tov import solve_profile, solve_structure
from nmbackend.cooling.physics.native import inner_core_sums

model=MetaModelEOS(muons=True,s_high='power')

@jax.jit
def mass(L,h):
    table=model.eos_table({'L_sym':L})
    return solve_structure(table,h,tidal=False,rtol=1e-12,atol=1e-14).M_msun

@jax.jit
def thermal(L,h):
    table=model.eos_table({'L_sym':L})
    comp=cooling_composition(model,{'L_sym':L})
    profile=solve_profile(table,h,n_points=2048,rtol=1e-12,atol=1e-14)
    return jnp.log(jnp.array(inner_core_sums(table,profile,h,comp,3e8,n_cut=.16)))

L0,h0=100.,.3
target=float(mass(L0,h0))
M_L,M_h=map(float,jax.jacfwd(mass,argnums=(0,1))(L0,h0))
assert np.isfinite(target) and M_h>0
h_L=-M_L/M_h
partial_L,partial_h=map(np.asarray,jax.jacfwd(thermal,argnums=(0,1))(L0,h0))
fixed=partial_L+partial_h*h_L
assert np.all(np.isfinite(fixed))
rows=[]
comparisons=[]
for step in (.02,.01,.005):
    values=[]
    for L in (L0-step,L0+step):
        h=brentq(lambda h:float(mass(L,h))-target,.2,.4,xtol=1e-12)
        residual=float(mass(L,h))-target
        slope=float(jax.jacfwd(mass,argnums=1)(L,h))
        assert abs(residual)<1e-8 and slope>0
        value=np.asarray(thermal(L,h))
        assert np.all(np.isfinite(value))
        values.append(value)
        rows.append({'L_sym':L,'h_c':h,'M_msun':float(mass(L,h)),'mass_residual_msun':residual,'dM_dh_c':slope})
    fd=(values[1]-values[0])/(2*step)
    assert np.all(np.isfinite(fd))
    comparison={'step_MeV':step,'fixed_mass_AD':fixed.tolist(),'fixed_mass_FD':fd.tolist()}
    comparisons.append(comparison)
    print(json.dumps(comparison),flush=True)
    np.testing.assert_allclose(fixed,fd,rtol=2e-3,atol=1e-5)
result={'target_mass_msun':target,'h_c_baseline':h0,'L_sym_baseline_MeV':L0,'T_inf_K':3e8,'n_cut_fm-3':.16,
        'quantities':['C','L_nu','L_durca','volume_cm3'],'derivative_units':'d ln(quantity) / d L_sym (MeV^-1)',
        'fixed_h_derivative':partial_L.tolist(),'partial_h_derivative':partial_h.tolist(),'dh_c_dL_sym_at_fixed_mass':h_L,
        'fixed_mass_derivative':fixed.tolist(),'comparisons':comparisons,'models':rows}
Path('fixed-mass-results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
