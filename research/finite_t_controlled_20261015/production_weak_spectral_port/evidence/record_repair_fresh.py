from pathlib import Path
import sys,json,dataclasses
R=Path(__file__).resolve().parents[1];which=sys.argv[1]
if which not in('old','new'):raise ValueError('Named frozen/current source only')
base=R/'pre_independent_record_repair/author'if which=='old'else R
sys.path.insert(0,str(base/'test_mirror/src'))
from nmbackend.reference import weak_reference as W
from nmbackend.reference import _weak_spectral_common as S
assert Path(W.__file__).resolve()==(base/'test_mirror/src/nmbackend/reference/weak_reference.py').resolve()
b=W.FreeElectronBath(*S.FIXED_BATH);rad=W.Radiation.fd(2.,b.equilibrium_mu_nue-.8,b.equilibrium_mu_antinue-.8)
r=W.reconstruct_spectral_moments('NEUTRON_DECAY',b,rad,W.EnergyRule.gauss(.3,.5,2),W.SpectralControls())
# One direct moving-threshold point also exercises positive/empty support paths.
p=W.capture_differential_coefficients('PC',b,1.7,.5987748936352157,-.1995027501115606,W.SpectralControls())
e=W.capture_differential_coefficients('PC',b,4.61,5.897,.99987,W.SpectralControls())
out=dict(reconstruction=dataclasses.asdict(r),positive_differential=dataclasses.asdict(p),genuine_empty=dataclasses.asdict(e))
(R/'evidence'/f'independent_record_repair_fresh_{which}.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(which,'complete')
