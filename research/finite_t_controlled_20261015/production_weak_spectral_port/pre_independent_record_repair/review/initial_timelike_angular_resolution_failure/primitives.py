"""Selected immutable pure definitions ONLY; never execute upstream top level."""
from pathlib import Path
from types import ModuleType,SimpleNamespace
import ast,copy,sys,json,hashlib,math,time
from dataclasses import dataclass
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import expit,log_expit,logsumexp
P=Path(__file__).resolve().parent;R=P.parent;LOCK=json.loads((P/'LOCK.json').read_text());MN,MP,ME,MW,HC,HBAR=[LOCK['masses'][n]for n in['MN','MP','ME','MW','HC','HBAR']];bath=SimpleNamespace(**LOCK['bath'])
for n,h in LOCK['sourcepins'].items():
 if hashlib.sha256((R/n).read_bytes()).hexdigest()!=h:raise RuntimeError('Pinned scientific source changed '+n)
def selected(path,names):return [copy.deepcopy(n)for n in ast.parse((R/path).read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef))and n.name in names]
def mod(name,body,env):
 m=ModuleType(name);m.__file__=str(P/(name+'.py'));sys.modules[name]=m;m.__dict__.update(dict(np=np,math=math,time=time,leggauss=leggauss,expit=expit,log_expit=log_expit,logsumexp=logsumexp,dataclass=dataclass));m.__dict__.update(env);tree=ast.fix_missing_locations(ast.Module(body=body,type_ignores=[]));(P/(name+'.py')).write_text('# ASTselected puredefinitions, injectedglobalsinprimitives.py; no upstreamdriver\n'+ast.unparse(tree)+'\n');exec(compile(tree,m.__file__,'exec'),m.__dict__);return m
et=ast.parse((R/'relativistic_capture_oracle/events.py').read_text());body=[copy.deepcopy(n)for n in et.body if isinstance(n,ast.Assign)and all(isinstance(t,ast.Name)and t.id in{'GF','VUD','HBAR','HC','S','Z','G','G5','I','MET'}for t in n.targets)or isinstance(n,ast.FunctionDef)and n.name in['adj','spinors','vertices','boost']];E=mod('spectral_review_event_current',body,{})
EC=mod('spectral_review_EC_source_scalar',selected('hot_electron_capture_moments/electron_capture.py',['dot','scalar_amplitude']),dict(E=E,MN=MN,MP=MP,ME=ME,MW=MW))
PC=mod('spectral_review_PC_source_scalar',selected('review_pc_moments/reference.py',['dot','amplitude']),dict(MN=MN,MP=MP,ME=ME,MW=MW,GF=1.1663785e-11,VUD=.974))
# Original pz-shell explicitspinors: constants copied independentofnewcandidate.
tree=ast.parse((R/'relativistic_delta_oracle/oracle.py').read_text());ob=[]
for n in tree.body:
 if isinstance(n,ast.Assign)and all(isinstance(t,ast.Name)and t.id in{'GF','VUD','HBAR','S','Z','G','G5','I','MET'}for t in n.targets)or isinstance(n,(ast.ClassDef,ast.FunctionDef))and n.name in['Point','adj','spinors','currents','setup','shell','integrate']:ob.append(copy.deepcopy(n))
O=mod('spectral_review_original_spinor_pz',ob,{})
# Alternate cosstar-delta reference is unchanged except explicit matchedbath injection.
bbody=selected('review_hot_beta_opacity_spectrum/reference.py',['slash','trace_amp','integral']);B=mod('spectral_review_beta_cosdelta',bbody,dict(E=E,MN=MN,MP=MP,ME=ME,HC=HC,HBAR=HBAR,ENDPOINT=(MN-MP-ME)*(MN+MP+ME)/(2*MN),S=SimpleNamespace(matched=lambda:(bath,None))))
proof={}
for tag,path,body,names in[('EC','hot_electron_capture_moments/electron_capture.py',selected('hot_electron_capture_moments/electron_capture.py',['dot','scalar_amplitude']),['dot','scalar_amplitude']),('PC','review_pc_moments/reference.py',selected('review_pc_moments/reference.py',['dot','amplitude']),['dot','amplitude']),('BETA','review_hot_beta_opacity_spectrum/reference.py',bbody,['slash','trace_amp','integral'])]:
 original={n.name:ast.dump(n,include_attributes=False)for n in selected(path,names)};proof[tag]={n.name:ast.dump(n,include_attributes=False)==original[n.name]for n in body}
assert all(all(v.values())for v in proof.values());(P/'EXTRACTION_PROVENANCE.json').write_text(json.dumps(dict(sourcepins=LOCK['sourcepins'],unchanged_selected_function_AST=proof,original_top_level_executed=False,beta_adaptation='OnlyinjectedS.matched returns lockedexplicitbath+None; no EOS/factory imports/calls or functionbodychanges',candidate_imported=False),indent=2)+'\n')
