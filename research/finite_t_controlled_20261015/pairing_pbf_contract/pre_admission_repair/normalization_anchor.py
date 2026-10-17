"""Independent natural-unit to cgs normalization of the sourced reduced formula."""
from pathlib import Path
import math,json
GF=1.1663785e-11;HC=197.3269804;HBAR=6.582119569e-22
KB=8.617333262145e-11;MEV_ERG=1.602176634e-6;MREF=940.
# Natural emissivity dimension E^5. fm^-3→cm^-3 is1e39.
conversion=MEV_ERG*1e39/(HC**3*HBAR)
coefficient=4*GF**2*MREF**2*(KB*1e9)**7/(15*math.pi**5)*conversion
error=coefficient/1.170e21-1
out=dict(formula='Q=4 GF² mL pF T^7/(15 pi^5) Nnu a F; Qnatural[MeV^5] -> cgs by MeVerg*1e39/(HC^3*hbar)',constants=dict(GF_MeV_m2=GF,HC_MeV_fm=HC,hbar_MeV_s=HBAR,kB_MeV_K=KB,MeV_erg=MEV_ERG,Mref_MeV=MREF),derived_coefficient=coefficient,legacy_rounded_coefficient=1.170e21,relative_difference=error,checks=[dict(name='source rounded coefficient vs independent cgs conversion',tolerance=.01,passed=abs(error)<.01)],qualification='Rounded constant check only; does NOT validate weak vertex approximation or total physical PBF accuracy. Legacy kF*197 and mass940 remain untouched.')
(Path(__file__).resolve().parent/'normalization_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
