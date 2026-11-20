from pathlib import Path
import sys,json,time
sys.dont_write_bytecode=True
from kernel import *
H=Path(__file__).parent
bath,m=matched_npe(.03,1e-12,.3)
d=differential(bath,.3,.993,0.,energy_order=8)
print('DIFF',d['j'],d['chi'])
r=spectral(bath,.3,angle_order=6,lepton_order=8,energy_order=8,widths=9.)
(H/'pilot.json').write_text(json.dumps(dict(matter=m,spectrum=r),indent=2)+'\n');print(r['j'],r['chi'],r['runtime_seconds'])
