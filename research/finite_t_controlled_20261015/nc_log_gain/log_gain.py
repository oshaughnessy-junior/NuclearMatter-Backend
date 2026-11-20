"""Log-domain NC gain/loss: combine occupation, DOS and response before exp."""
from pathlib import Path
import sys,math
import numpy as np
from scipy.special import log_expit,logsumexp
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'nc_continuum_grid'))
import redistribution as original
nc=original.nc
Radiation=original.Radiation
FIELDS=['out','transport','energy_positive','energy_negative','gain','loss','net_positive','net_negative']

def represented(log,sign=1):
 if sign not in [-1,0,1]:raise ValueError('Sign must be -1,0,1')
 if sign==0:
  if log is not None:raise ValueError('Exact zero must have log=None')
  return dict(sign=0,log_abs=None,value=0.,representable=True,reason='exact_zero')
 if log is None or not math.isfinite(log):raise ArithmeticError('Nonzero result requires finite logarithm')
 try:value=math.exp(log)
 except OverflowError:return dict(sign=sign,log_abs=float(log),value=None,representable=False,reason='overflow')
 return dict(sign=sign,log_abs=float(log),value=sign*value if value else None,representable=value>0,reason=None if value else 'underflow')

def difference(a,b):
 """Signed log(exp(a)-exp(b)); None denotes an algebraic empty/zero term."""
 if any(v is not None and not math.isfinite(v)for v in [a,b]):raise ArithmeticError('Finite logs required')
 if a is None:return (0,None)if b is None else(-1,b)
 if b is None:return 1,a
 if not all(math.isfinite(x)for x in [a,b]):raise ArithmeticError('Finite logs required')
 if a==b:return 0,None
 if a>b:return 1,a+math.log(-math.expm1(b-a))
 return -1,b+math.log(-math.expm1(a-b))

def occupations(rad,E):
 x=(rad.mu-E)/rad.T
 if not math.isfinite(x):raise ArithmeticError('Radiation logit outside finite represented domain')
 lf=float(log_expit(x));lb=float(log_expit(-x))
 if rad.kind=='FD':return lf,lb
 lf=float(logsumexp([math.log(.15)+lf,math.log(.45)-((E-14.)/3.)**2]))
 return lf,math.log1p(-math.exp(lf))

def point(species,E,Ep,c,rad,*,reverse='kms'):
 original.controls(E,8)
 if not all(math.isfinite(x)for x in [Ep,c])or Ep<=0 or not -1<=c<=1:raise ValueError('Finite Ep>0 and c in[-1,1] required')
 if reverse not in ['kms','direct']:raise ValueError('Named kms/direct reverse path required')
 pair=nc.pair_from_species(species);omega=E-Ep
 forward=nc.log_kernel(pair,E,Ep,c);lf,lb=occupations(rad,Ep);lfi,lbi=occupations(rad,E)
 gain=(forward-omega/species.T if reverse=='kms'else nc.log_kernel(pair,Ep,E,c)+2*math.log(Ep/E))+lf+lbi
 loss=forward+lfi+lb
 if reverse=='kms'and rad.kind=='FD':
  delta=omega*(1/rad.T-1/species.T)
  if delta==0:sign,net=0,None
  elif delta>0:sign,net=1,loss+delta+math.log(-math.expm1(-delta))
  else:sign,net=-1,loss+math.log(-math.expm1(delta))
 else:sign,net=difference(gain,loss)
 return dict(log_forward_kernel=forward,log_gain=gain,log_loss=loss,log_blocked_out=forward+lb,net_sign=sign,log_abs_net=net,gain=represented(gain),loss=represented(loss),net=represented(net,sign),reverse=reverse)

def angular_logs(species,E,Ep,rad,order,reverse):
 pair=nc.pair_from_species(species);xs,ws=original.nodes_cached(order);lo,hi=abs(E-Ep),E+Ep
 if hi<=lo:raise ArithmeticError('q interval collapsed')
 qs=lo+(xs+1)*(hi-lo)/2;weights=ws*(hi-lo)/2*qs/(E*Ep);cs=(E*E+Ep*Ep-qs*qs)/(2*E*Ep)
 lf,lb=occupations(rad,Ep);lfi,lbi=occupations(rad,E);omega=E-Ep;terms={k:[]for k in FIELDS}
 for q,c,w in zip(qs,cs,weights):
  c=max(-1.,min(1.,float(c)));W=nc.angular_weight(c,nc.NEUTRON)
  forward=math.log(nc.K)+2*math.log(Ep)+math.log(W)+original.log_response(pair,float(q),omega)
  logw=math.log(float(w));out=forward+lb
  gain=(forward-omega/species.T if reverse=='kms'else math.log(nc.K)+2*math.log(E)+math.log(W)+original.log_response(pair,float(q),-omega)+2*math.log(Ep/E))+lf+lbi
  loss=forward+lfi+lb
  terms['out'].append(logw+out);terms['transport'].append(logw+out+math.log1p(-c));terms['gain'].append(logw+gain);terms['loss'].append(logw+loss)
  if omega:terms['energy_positive'if omega>0 else'energy_negative'].append(logw+out+math.log(abs(omega)))
  if reverse=='kms'and rad.kind=='FD':
   delta=omega*(1/rad.T-1/species.T)
   if delta>0:sign,net=1,loss+delta+math.log(-math.expm1(-delta))
   elif delta<0:sign,net=-1,loss+math.log(-math.expm1(delta))
   else:sign,net=0,None
  else:sign,net=difference(gain,loss)
  if sign:terms['net_positive'if sign>0 else'net_negative'].append(logw+net)
 return {key:float(logsumexp(vals))if vals else None for key,vals in terms.items()}

def integrate(species,E,rad,*,energy_order=64,angle_order=192,cutoff=None,tail=70.,reverse='kms'):
 original.controls(E,energy_order);original.controls(E,angle_order)
 if reverse not in ['kms','direct']:raise ValueError('Named kms/direct reverse path required')
 default=original.upper(species,E,tail);cutoff=default if cutoff is None else cutoff
 if not math.isfinite(cutoff)or cutoff<=E:raise ValueError('Finite cutoff>Ein required')
 es=original.edges(species,E,cutoff);xs,ws=original.nodes_cached(energy_order);terms={key:[]for key in FIELDS};panels=[]
 for lo,hi in zip(es[:-1],es[1:]):
  partial={key:[]for key in FIELDS}
  for x,w in zip(xs,ws):
   Ep=float(lo+(x+1)*(hi-lo)/2);lw=math.log(float(w*(hi-lo)/2));vals=angular_logs(species,E,Ep,rad,angle_order,reverse)
   for key,v in vals.items():
    if v is not None:partial[key].append(lw+v)
  logp={key:float(logsumexp(v))if v else None for key,v in partial.items()};panels.append(dict(lo=lo,hi=hi,logs=logp))
  for key,v in logp.items():
   if v is not None:terms[key].append(v)
 logs={key:float(logsumexp(v))if v else None for key,v in terms.items()}
 esign,elog=difference(logs['energy_positive'],logs['energy_negative']);nsign,nlog=difference(logs['net_positive'],logs['net_negative']);absenergy=float(logsumexp([logs['energy_positive'],logs['energy_negative']]))
 moments=[represented(logs['out']),represented(logs['transport']),represented(elog,esign),represented(absenergy),represented(logs['gain']),represented(logs['loss']),represented(nlog,nsign)]
 return dict(values=[x['value']for x in moments],moments=moments,positive_component_logs=logs,panels=panels,cutoff=cutoff,energy_order=energy_order,angle_order=angle_order,reverse=reverse,labels=original.LABELS)
