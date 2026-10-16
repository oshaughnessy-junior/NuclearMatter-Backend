"""Independent neutron/proton particle states, not a composed neutral EOS."""
import json
from nmbackend.constants import M_N, M_P
from nmbackend.reference.finite_t_baryons import state_density

neutron = state_density(2., .001, mass_mev=M_N)
proton = state_density(2., .0002, mass_mev=M_P)
print(json.dumps({
    "model": "particle-only ideal gas; independent species, fixed supplied masses",
    "units": "T/mu/m MeV; particle n/s/Cv fm^-3; energy/pressure MeV fm^-3",
    "neutron": neutron.asdict(),
    "proton": proton.asdict(),
}, indent=2))
