"""Literal point V-A initial-spin-average4 amplitude, explicit deformed masses."""
import numpy as np
GF=1.1663785e-11;VUD=.974;GA=1.2723
def dot(a,b):return a[:,0]*b[:,0]-np.sum(a[:,1:]*b[:,1:],axis=1)
def amplitude(k,pa,l,pb,ma,mb,anti=False):
 ka,kb,la,lb,kl,ab=[dot(x,y)for x,y in[(k,pa),(k,pb),(l,pa),(l,pb),(k,l),(pa,pb)]]
 vv=ka*lb+kb*la-ma*mb*kl;aa=GA*GA*(ka*lb+kb*la+ma*mb*kl);va=(-1 if anti else 1)*2*GA*(ka*lb-kb*la)
 return 8*GF*GF*VUD*VUD*(vv+aa+va)
def vertices(q,M,E):
 # M is the recorded bare weak normalization; no tensor/pseudoscalar term
 # remains in a literal point V-A current, so the matrix is independent of it.
 return np.broadcast_to(E.G-GA*(E.G@E.G5),(len(q),4,4,4)).copy()
