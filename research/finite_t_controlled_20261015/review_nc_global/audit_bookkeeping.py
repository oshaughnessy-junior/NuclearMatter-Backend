from pathlib import Path
import json,math
P=Path(__file__).resolve().parent;r=json.loads((P/'results.json').read_text());fine=r['raw']['independent_low_fine']['values'];pair=r['raw']['paired_low_reference']['values'];hc=197.3269804
checks=[]
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=math.isfinite(error)and error<tol))
# Explicit solid-angle spin-one-helicity derivation; no second outgoing DOS.
derived=4*math.pi/(2*math.pi*hc)**3;stored=1/(2*math.pi**2*hc**3)
gate('isotropic one-helicity DOS from solid angle',abs(derived/stored-1),1e-14)
for k,name in [(5,'forward'),(6,'reverse')]:gate('directpositive event scale vs halftriangle sum '+name,abs(fine[k]/(pair[5]+pair[6])-1),3e-5)
gate('finite-number FD entropy chemical sign',abs(fine[2]+fine[3]/2.35+2.8*fine[0]/2.35)/abs(fine[2]),1e-12)
report=dict(passed=all(c['passed']for c in checks),checks=checks,derived_DOS=derived,scope='Measure/gross-event/FD entropy bookkeeping; does not add physical quadrature')
(P/'bookkeeping.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
