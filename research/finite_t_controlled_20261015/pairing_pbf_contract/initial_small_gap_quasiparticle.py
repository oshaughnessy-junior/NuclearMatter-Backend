"""Reduced BCS quasiparticle energy/angular integral, no weak vertex tensor."""
import math,numbers
import numpy as np
from scipy.integrate import quad

def log_F_A(v):
    """log[v² int_0^inf dx z^4/(1+exp(z))²], z=hypot(x,v)."""
    if isinstance(v,bool) or not isinstance(v,numbers.Real) or not math.isfinite(v) or not 0<=v<=1000:raise ValueError('finite0<=v<=1000 required')
    if v==0:return -math.inf
    scale=math.sqrt(max(v,1.))
    def fun(t):
        x=scale*t;z=math.hypot(x,v);difference=x*x/(z+v)
        return scale*z**4*math.exp(-2*difference)/(1+math.exp(-z))**2
    value=quad(fun,0,math.inf,epsabs=2e-10,epsrel=3e-12,limit=200)[0]
    return 2*math.log(v)-2*v+math.log(value)

def log_F_B(v):
    """Angular average int_0^1 dmu F_A(v sqrt(1+3mu²)).

    v is minimum gap/T, not rms gap/T; exact stated BCS angular function.
    """
    if isinstance(v,bool) or not isinstance(v,numbers.Real) or not math.isfinite(v) or not 0<=v<=500:raise ValueError('finite0<=v<=500 required')
    if v==0:return -math.inf
    scale=math.sqrt(max(v,1.))
    def fun(t):
        mu=t/scale;u=v*math.sqrt(1+3*mu*mu)
        return math.exp(log_F_A(u)+2*v)/scale
    value=quad(fun,0,scale,epsabs=2e-10,epsrel=2e-10,limit=120)[0]
    return -2*v+math.log(value)

def represented(log_value):
    if log_value==-math.inf:return dict(log_value=None,value=0.,status='exact_zero')
    if not math.isfinite(log_value):raise ValueError('finite log value or exact−infinity zero required')
    if log_value<math.log(float(np.nextafter(0.,1.))):return dict(log_value=log_value,value=None,status='positive_below_float64_representation')
    return dict(log_value=log_value,value=math.exp(log_value),status='positive_represented')
