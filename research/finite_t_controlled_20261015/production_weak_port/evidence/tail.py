from pathlib import Path
import sys,json,dataclasses,time
P=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True;sys.path.insert(0,str(P/'test_mirror/src'))
from nmbackend.reference import weak_reference as W
lock=json.loads((P/'LOCK.json').read_text());channel=sys.argv[1];b=W.FreeElectronBath(**lock['heldouts'][1]);mu,anti=b.equilibrium_mu_nue,b.equilibrium_mu_antinue
# Isolated actual finite-parent/charged-lepton tail at the second heldout.
# Same fine axis counts/maps and both FD radiation affinities plus empty.
rows={};start=time.monotonic()
for name,rad in [('empty',W.Radiation.empty()),('emission',W.Radiation.fd(b.T,mu-.6*b.T,anti-.8*b.T)),('inverse',W.Radiation.fd(b.T,mu+.6*b.T,anti+.8*b.T))]:
 c=dataclasses.replace(W.Controls.fine(channel),tail=lock['isolated_tail'][channel]);rows[name]=dataclasses.asdict(W.channel_moments(channel,b,rad,c))
(P/'evidence'/f'tail_{channel}.json').write_text(json.dumps(dict(channel=channel,bath=dataclasses.asdict(b),rows=rows,seconds=time.monotonic()-start),indent=2,allow_nan=False)+'\n');print(channel,time.monotonic()-start)
