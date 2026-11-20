"""Small input and ordinary-representation contracts; no rate evaluation."""
from pathlib import Path
import json, math
import log_emission as r
checks=[]
def check(name,ok):checks.append(dict(name=name,passed=bool(ok)))
for x,expected in [(-1000.,False),(-745.,True),(-744.,True),(0.,True),(710.,False)]:
    a=r.represented(x)
    check(f'representation {x}',a['representable']==expected and (a['value'] is None if not expected else a['value']==math.exp(x)))
for x in [math.nan,math.inf,-math.inf]:
    try:r.represented(x)
    except ArithmeticError:check(f'nonfinite representation {x}',True)
    else:check(f'nonfinite representation {x}',False)
bath=r.make_bath(.3,7e-5,3e-5,1e7,0.)
for name,kw in [('bool order',dict(outer_order=True)),('fractional order',dict(outer_order=8.5)),('NaN tail',dict(tail=math.nan)),('negative tail',dict(tail=-1)),('unsorted edges',dict(edges=[0,2,1])),('NaN edge',dict(edges=[0,math.nan])),('nonzero first edge',dict(edges=[1,2])),('negative runtime',dict(max_seconds=-1))]:
    try:r.log_moments(bath,**kw)
    except ValueError:check(name,True)
    else:check(name,False)
report=dict(status='PASS'if all(c['passed']for c in checks)else'FAIL',checks=checks)
Path(__file__).with_name('controls.json').write_text(json.dumps(report,indent=2)+'\n')
print(report)
assert report['status']=='PASS'
