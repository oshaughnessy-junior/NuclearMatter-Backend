"""Distinguish smallest-normal gap input from legitimate ordinary-output underflow."""
from pathlib import Path
import importlib.util,sys,math,json
sys.dont_write_bytecode=True
p=Path(__file__).resolve().parent;a=p.parent/'pairing_pbf_contract/quasiparticle.py'
s=importlib.util.spec_from_file_location('pbf_boundary_only',a);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
v=sys.float_info.min;la=m.log_F_A(v);lb=m.log_F_B(v);err=math.expm1(lb-la-math.log(2))
d=dict(v=v,smallest_positive_normal_binary64=True,logFA=la,logFB=lb,FB_over_FA_minus2_relative=err,FA=m.represented(la),FB=m.represented(lb),checks=[dict(name='smallestNORMALinput angular coefficient',passed=abs(err)<3e-10,error=err,tolerance=3e-10),dict(name='normalINPUT with ordinaryOUTPUT underflow admitted',passed=math.isfinite(lb)and m.represented(lb)['value']is None)])
(p/'boundary_results.json').write_text(json.dumps(d,indent=2)+'\n')
