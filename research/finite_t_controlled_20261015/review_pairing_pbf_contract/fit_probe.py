from pathlib import Path
import importlib.util,sys,json,math
P=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('review_pbf_extra',P/'probe.py');m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
d=json.loads((P/'results.json').read_text());checks=[];rows=[]
for row in d['new_integral_points']:
 v=row['v']
 if not .01<v<=300:continue
 for ch,fun,logkey in [('A',m.C.LEGACY['_ctrl_1s0'],'independent_logFA'),('B',m.C.LEGACY['_ctrl_3p2b'],'independent_logFB')]:
  actual=float(fun(v));err=math.expm1(math.log(actual)-row[logkey]);rows.append(dict(channel=ch,v=v,fit=actual,relative_error=err));checks.append(dict(name='freshfit '+ch+str(v),error=err,tolerance=.05,passed=abs(err)<=.05))
out=dict(checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],rows=rows);(P/'fit_results.json').write_text(json.dumps(out,indent=2)+'\n');print({k:out[k]for k in ['passed','failed']})
