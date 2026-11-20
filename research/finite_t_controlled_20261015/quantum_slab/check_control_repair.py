"""Regression for independently discovered fallback/control bypass."""
import json
from pathlib import Path
import fixed_bath as f
import guarded_capture as g


def main():
    matter=f.matter_state(2,.002,.001)
    optical=matter.optical('nue')
    checks=[]
    for kw in ({'order':72,'tail':-1.},{'order':True},{'order':4.5},{'tail':float('nan')}):
        for mode in ('wrapper','direct_guard'):
            try:
                if mode=='wrapper':
                    f.coefficients(matter,'nue',60,**kw)
                else:
                    n=kw.get('order',48)
                    g.log_coefficients(optical,60,energy_order=n,angle_order=n,tail=kw.get('tail',40))
                passed=False
            except ValueError:
                passed=True
            checks.append(dict(case=str(kw),mode=mode,passed=passed))
    valid=f.coefficients(matter,'nue',60,order=72,tail=50)
    checks.append(dict(case='valid_witness_still_uses_guard',passed=valid['capture_panel_guard']['used']))
    result=dict(passed=all(c['passed'] for c in checks),checks=checks,valid_witness=valid)
    Path(__file__).with_name('control_repair_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=result['passed'],checks=len(checks))))
    if not result['passed']:raise SystemExit(1)

if __name__=='__main__':main()
