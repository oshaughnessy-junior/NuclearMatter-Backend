"""Quantify the mild selected physical coefficient variation; no fit."""
from pathlib import Path
import sys,json
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'conductive_thermal_slab'));import slab
p=slab.Physics();rows=[]
for theta in[.96,1.,1.04]:rows.append(dict(theta=theta,kappa_ratio=p.k(theta),Cv_ratio=p.cv(theta),diffusivity_ratio=p.k(theta)/p.cv(theta),W=p.W(theta)))
Path(__file__).with_name('coefficient_range.json').write_text(json.dumps(dict(rows=rows,scope='Actual source coefficients at selected temperature extrema; mild nonlinearity, not broader domain qualification'),indent=2)+'\n');print(rows)
