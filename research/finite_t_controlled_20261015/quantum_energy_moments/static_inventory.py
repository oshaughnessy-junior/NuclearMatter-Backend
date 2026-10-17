"""One complete antineutrino inventory: heavy-M integrated number/energy limit."""
from integrated import *
from scipy.integrate import quad
import static_capture as static
import json,time
P=Path(__file__).resolve().parent;start=time.monotonic()
m=slab.matter_state(2.,7e-5,3e-5,mass=1e7)
s=static.State(2.,m.electron.chemical_potential,7e-5,3e-5)
Q=1.29333236;ME=.51099895
edges=(0.,.1,.5,Q-ME,Q+ME,2.,4.,8.,16.,24.,40.,70.,120.)
r=integrate(m,'antinue',outer_order=12,internal_order=72,edges=edges)
channels={};expected=np.zeros(2)
for channel in ('positron_capture','neutron_decay'):
 vals=[]
 for power in (2,3):
  val=sum(quad(lambda E:E**power*static.spectral(s,E,channel)[0],lo,hi,epsabs=1e-18,epsrel=2e-11)[0] for lo,hi in zip(edges[:-1],edges[1:]))*slab.DOS*1e39
  vals.append(val)
 channels[channel]=vals;expected+=vals
actual=np.array([r['values']['emission_number_cm3_s'],r['values']['emission_energy_MeV_cm3_s']]);errors=abs(actual/expected-1)
report=dict(passed=bool(np.all(errors<3e-5)),relative_errors=errors.tolist(),dynamic=actual.tolist(),static=expected.tolist(),static_channels_number_energy=channels,heavy_mass_MeV=1e7,quadrature=r,runtime_seconds=time.monotonic()-start,scope='Mathematical heavy-M dilute classical limit at T2, antineutrino complete positron capture+neutron decay inventory; not a physical cold-state benchmark')
(P/'static_inventory_results.json').write_text(json.dumps(report,indent=2)+'\n');print('static inventory',report['passed'],errors,report['runtime_seconds'])
