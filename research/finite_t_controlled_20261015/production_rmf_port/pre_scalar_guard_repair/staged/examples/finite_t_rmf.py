"""Two locally admitted nuclear states and one explicit unadmitted spinodal."""
import json
from nmbackend.reference.finite_t_rmf import state_density, diagnose_density

states=[state_density(10.,.16,.1),state_density(20.,.24,.3)]
spinodal=diagnose_density(1.,.06,.1)
print(json.dumps({
    'model':'exact rounded GM1, common939MeV, nuclear-only no-sea thermal pairs',
    'units':'T,fields,mD,nu,mu MeV; n,s,Cv fm^-3; e,P,f MeV fm^-3; H MeV fm^3',
    'states':[s.asdict()for s in states],
    'diagnostic_only_unadmitted':spinodal.asdict(),
    'scope':'local nuclear stability only; no lepton/beta/phase coexistence or JAX qualification',
},indent=2))
