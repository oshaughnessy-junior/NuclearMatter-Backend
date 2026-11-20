from pathlib import Path
import json,math,numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import kve
P=Path(__file__).parent;rows=[]
for z in [40.,100.,300.,600.]:
 values=[]
 for N in [64,96]:
  x,w=leggauss(N);upper=math.sqrt(180*(1+45/z));x=(x+1)*upper/2;w=w*upper/2
  # E-m=m*x²/[z*(sqrt(1+x²/z)+1)] avoids cold subtraction.
  kt=x*x/(np.sqrt(1+x*x/z)+1);v=float(sum(w*x*x*np.exp(-kt)/(1+np.exp(-z-kt))));values.append(v)
 want=math.sqrt(z)*float(kve(2,z));rows.append(dict(z=z,dimensionless_gauss=values,scaled_bessel=want,refinement=values[0]/values[1]-1,error=values[1]/want-1,passed=bool(abs(values[1]/want-1)<3e-12)))
out=dict(rows=rows,passed=sum(r['passed']for r in rows),failed=[r for r in rows if not r['passed']]);(P/'scaled_dilute_anchor.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
