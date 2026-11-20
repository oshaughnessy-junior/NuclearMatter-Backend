from pathlib import Path
import sys,json,dataclasses,time
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'test_mirror/src'))
from nmbackend.reference import weak_reference as W
b=W.FreeElectronBath(**json.loads((R/'LOCK.json').read_text())['bath'])
channel=sys.argv[1];order=int(sys.argv[2]);name=sys.argv[3]
lo,hi=(.2,.8)if channel=='NEUTRON_DECAY'else(2.,6.)
mu,anti=b.equilibrium_mu_nue,b.equilibrium_mu_antinue
rad={'empty':W.Radiation.empty(),'LTE':W.Radiation.fd(2.,mu,anti),'noneq':W.Radiation.fd(2.,mu-.8,anti-.8)}[name]
start=time.monotonic()
try:
 row=W.reconstruct_spectral_moments(channel,b,rad,W.EnergyRule.gauss(lo,hi,order),W.SpectralControls())
 out=dict(result=dataclasses.asdict(row),seconds=time.monotonic()-start)
except Exception as err:out=dict(error=type(err).__name__,message=str(err),seconds=time.monotonic()-start)
(R/'evidence'/f'pilot_{channel}_{order}_{name}.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(channel,order,name,out.get('error','PASS'),out['seconds'])
if 'error'in out:sys.exit(1)
