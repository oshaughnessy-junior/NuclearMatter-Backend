"""Produce owned current functions by one literal F2-assignment substitution."""
from pathlib import Path
import ast,copy,hashlib,json
P=Path(__file__).resolve().parent;ROOT=P.parent;lock=json.loads((P/'LOCK.json').read_text());items=[];definitions=[]
class Config(ast.NodeTransformer):
 def __init__(self,off):self.off=off;self.count=0
 def visit_Assign(self,node):
  if len(node.targets)==1 and isinstance(node.targets[0],ast.Name)and node.targets[0].id=='f':
   self.count+=1
   if self.off:node.value=ast.Constant(0.)
  return self.generic_visit(node)
 def visit_Attribute(self,node):
  if isinstance(node.value,ast.Name)and node.value.id=='E'and node.attr in ['GF','VUD']:return ast.Name(node.attr,ast.Load())
  return self.generic_visit(node)
for channel,path,name in [('EC','hot_electron_capture_moments/electron_capture.py','scalar_amplitude'),('PC','review_pc_moments/reference.py','amplitude')]:
 src=(ROOT/path).read_bytes();assert hashlib.sha256(src).hexdigest()==lock['pins'][path];fn=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef)and n.name==name)
 for variant,off in [('full',False),('F2zero',True)]:
  transform=Config(off);new=transform.visit(copy.deepcopy(fn));assert transform.count==1;new.name=channel.lower()+'_'+variant;ast.fix_missing_locations(new);definitions.append(ast.unparse(new));items.append(dict(name=new.name,original_source=path,original_sha256=lock['pins'][path],f2_assignment_count=transform.count,change='f=0 ONLY'if off else'No physical change; E.GF/E.VUD bound to equal literal constants'))
prefix='"""Owned configured currents; generated from pinned source functions, not a production kernel."""\nimport numpy as np\nMN=939.56542052;MP=938.27208816;ME=.51099895069;MW=(MN+MP)/2;GF=1.1663785e-11;VUD=.974\ndef dot(a,b):return a[:,0]*b[:,0]-np.sum(a[:,1:]*b[:,1:],axis=1)\n'
assert not(P/'configured_currents.py').exists();(P/'configured_currents.py').write_text(prefix+'\n\n'.join(definitions)+'\n');(P/'CONFIGURATION_PROVENANCE.json').write_text(json.dumps(dict(functions=items,generated_source_sha256=hashlib.sha256((P/'configured_currents.py').read_bytes()).hexdigest()),indent=2)+'\n')
