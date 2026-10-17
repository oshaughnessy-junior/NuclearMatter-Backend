from pathlib import Path
import sys,json
sys.dont_write_bytecode=True
from slab import *
p=Physics();x=(np.arange(8)+.5)/8;t=1+.12*np.cos(math.pi*x)
r=evolve(p,t,.02,16)
Path(__file__).with_name('pilot.json').write_text(json.dumps(dict(result=r,scales=p.scales()),indent=2)+'\n');print(r['runtime_seconds'],r['EOS_calls'],r['transport_calls'],p.scales())
