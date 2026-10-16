"""Scoped normal-input admission repair; no legacy/source physics replay."""
from pathlib import Path
import ast,hashlib,importlib.util,json,math,sys,time,warnings
import numpy as np
from scipy.special import zeta
P=Path(__file__).resolve().parent;SNAP=P/'pre_admission_repair'
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    start=time.monotonic();old=load('pbf_admission_reviewed_old',SNAP/'quasiparticle.py');new=load('pbf_admission_repaired',P/'quasiparticle.py');checks=[];rejected=[];rows=[];old_failures=[]
    def check(n,ok):checks.append(dict(name=n,passed=bool(ok)))
    tiny=np.finfo(float).tiny;below=float(np.nextafter(tiny,0.));above=float(np.nextafter(tiny,math.inf));minimum=float(np.nextafter(0.,1.))
    # Exact old reviewer inputs and the immediate normal/subnormal boundary.
    for v in [minimum,1e-320,below]:
        for name in ['log_F_A','log_F_B']:
            try:getattr(new,name)(v);bad=False;message='accepted'
            except ValueError as e:bad=True;message=str(e)
            check('reject '+name+' '+v.hex(),bad and 'Subnormal' in message);rejected.append(dict(function=name,v=v,v_hex=v.hex(),rejected=bad,message=message))
    coefficient=24*((1-2**-4)*zeta(5)-(1-2**-3)*zeta(4))
    for v in [1e-320,minimum]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');b=old.log_F_B(v)
        error=math.expm1(b-(2*math.log(v)+math.log(2*coefficient)))
        old_failures.append(dict(v=v,v_hex=v.hex(),old_log_B=b,relative_small_gap_error=error,unchanged_old_gate=3e-10,old_physical_gate_pass=abs(error)<=3e-10,warnings=[str(x.message) for x in caught]))
        check('old reviewer error reproduced '+v.hex(),abs(error)>3e-10)
    for v in [0.,-0.,tiny,above,1e-307,1e-250,1e-200,.017,.43,2.7,37.,127.,499.]:
        row=dict(v=v,v_hex=v.hex())
        for name in ['log_F_A','log_F_B']:
            a=getattr(old,name)(v);b=getattr(new,name)(v)
            check('bitwise unchanged '+name+' '+v.hex(),a.hex()==b.hex())
            row[name]=dict(old=a if math.isfinite(a) else None,new=b if math.isfinite(b) else None,hex_old=a.hex(),hex_new=b.hex())
        rows.append(row)
    for v in [tiny,above]:
        error=new.log_F_B(v)-(2*math.log(v)+math.log(2*coefficient))
        check('normal boundary analytic angular coefficient '+v.hex(),abs(error)<3e-10)
        for name in ['log_F_A','log_F_B']:
            r=new.represented(getattr(new,name)(v));check('normal INPUT with unrepresented OUTPUT '+name+' '+v.hex(),r['value'] is None and r['status']=='positive_below_float64_representation' and math.isfinite(r['log_value']))
    outputs=[]
    for name,v in [('log_F_A',1000.),('log_F_B',500.)]:
        a=getattr(old,name)(v);b=getattr(new,name)(v);r=new.represented(b)
        check('deep output log bitwise unchanged '+name,a.hex()==b.hex());check('deep output positive/logNone '+name,r['value'] is None and r['status']=='positive_below_float64_representation' and math.isfinite(r['log_value']));outputs.append(dict(function=name,v=v,result=r))
    for name in ['log_F_A','log_F_B']:
        check('exact zero policy '+name,getattr(new,name)(0.)==-math.inf and new.represented(getattr(new,name)(0.))==old.represented(getattr(old,name)(0.)))
        for v in [-1.,float('inf'),float('nan'),True]:
            try:getattr(new,name)(v);ok=False
            except ValueError:ok=True
            check('existing invalid policy '+name+' '+str(v),ok)
    # Remove only the new admission if and changed docstrings for equation audit.
    def scientific_ast(path):
        tree=ast.parse(path.read_text())
        for f in [x for x in tree.body if isinstance(x,ast.FunctionDef)]:
            f.body=[x for x in f.body if not (isinstance(x,ast.Expr) and isinstance(x.value,ast.Constant) and isinstance(x.value.value,str)) and not (isinstance(x,ast.If) and ast.unparse(x.test)=='v < np.finfo(float).tiny')]
        return ast.dump(tree,include_attributes=False)
    check('all equation and other policy AST unchanged',scientific_ast(SNAP/'quasiparticle.py')==scientific_ast(P/'quasiparticle.py'))
    pins=json.loads((P/'PRE_ADMISSION_REPAIR_PIN.json').read_text())
    keep=['contract.py','source_snapshot/core_nu.py','source_snapshot/superfluid.py','results.json','alternate_integral_results.json','fortran_results.json','endpoint_results.json','normalization_results.json','initial_small_gap_log_failure.json','initial_small_gap_quasiparticle.py']
    for name in keep:check('science/raw unchanged '+name,sha(P/name)==pins[name])
    raw=json.loads((P/'results.json').read_text());check('four legacy endpoint exclusions unchanged',sum(not x['literal_physical_exact_zero_gate'] for x in raw['raw_endpoint_exclusions'])==4)
    out=dict(checks=checks,passed=sum(x['passed'] for x in checks),failed=[x for x in checks if not x['passed']],admission=dict(zero='exact zero remains allowed',positive='float64 normal input tiny<=v, with previous upper bounds A1000/B500',subnormal='reject0<v<np.finfo(float).tiny; observed angular rounding failure, shared A/B domain policy',output='finite log with positive unrepresented ordinaryNone remains allowed'),boundary=dict(tiny=tiny,below=below,above=above,minimum_positive=minimum),rejected=rejected,old_failures=old_failures,bitwise_rows=rows,deep_outputs=outputs,scientific_source_pin=sha(P/'contract.py'),repaired_helper_pin=sha(P/'quasiparticle.py'),seconds=time.monotonic()-start)
    (P/'admission_repair_results.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps({k:out[k] for k in ['passed','failed','seconds']}))

if __name__=='__main__':main()
