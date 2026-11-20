"""Direct EC/PC j and true chi in exact original baryon delta-shell phase."""
from __future__ import annotations

import math
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import logsumexp

from . import _weak_current as C
from ._weak_quadrature import log_fd
from ._weak_spectral_common import (SpectralResult, coefficient, empty_coefficient,
                                    differential_inputs, validate)


def support(ma, mb, omega, q):
    """All positive-energy solutions of the original polar-delta inequality.

    No discriminant clipping, physical width, or q=0 empty-support shortcut.
    A=q²−omega² uses a factored product near the exact linear boundary.
    """
    if q <= 0 or not all(math.isfinite(v) for v in (ma, mb, omega, q)):
        raise ArithmeticError('Unresolved zero/singular transfer coordinate')
    aa = (q-abs(omega))*(q+abs(omega))
    cc = (ma-mb)*(ma+mb)-aa
    bb = -omega*cc
    dd = -(cc*cc/4+q*q*ma*ma)
    low = max(ma, mb-omega)
    if aa == 0:
        if bb == 0:
            return ((low, None),) if dd >= 0 else ()
        edge = -dd/bb
        if bb > 0:
            return ((max(low, edge), None),)
        return ((low, edge),) if edge > low else ()
    disc = q*q*(cc*cc+4*aa*ma*ma)
    if not math.isfinite(disc):
        raise ArithmeticError('Unresolved support discriminant')
    if disc < 0:
        return ()
    if disc == 0:
        raise ArithmeticError('Degenerate support root requires a separately derived limit')
    root = math.sqrt(disc)
    stable = (-bb-math.copysign(root, bb))/(2*aa)
    other = dd/(aa*stable) if stable else (-bb+root)/(2*aa)
    r1, r2 = sorted((stable, other))
    if aa < 0:
        return ((max(low, r1), r2),) if r2 > max(low, r1) else ()
    result = []
    if r1 > low: result.append((low, r1))
    result.append((max(low, r2), None))
    return tuple(result)


def differential(channel, bath, energy, el, cosine, controls, timer):
    energy = validate(channel, bath, energy, controls)
    if channel not in ('EC', 'PC'):
        raise ValueError('Capture differential measure is not a decay measure')
    el, cosine = differential_inputs(el, cosine)
    timer.check()
    ma, mb = (C.MN, C.MP) if channel == 'EC' else (C.MP, C.MN)
    mua, mub = (bath.mu_n, bath.mu_p) if channel == 'EC' else (bath.mu_p, bath.mu_n)
    pe = math.sqrt((el-C.ME)*(el+C.ME)); omega = energy-el
    q = math.sqrt((energy-pe)**2+2*energy*pe*(1-cosine))
    intervals = support(ma, mb, omega, q)
    if not intervals:
        return SpectralResult(channel,bath,energy,empty_coefficient(),empty_coefficient(),controls,
                              ('analytic_empty_particle_hole_support',),(el,cosine),units='s^-1 MeV^-1 per cosine')
    xs, ws = leggauss(controls.energy_order); energies = []; weights = []; panels = []
    for lo, bound in intervals:
        hi = max(lo,ma,mua,mub-omega)+controls.tail*bath.T
        if bound is not None: hi = min(hi,bound)
        if not hi > lo:
            raise ArithmeticError('Positive support interval collapsed; no weighted omission bound')
        cuts = sorted({lo,hi}|{center+h*bath.T for center in (ma,mua,mub-omega)
                              for h in (0,2,6,12,24) if lo < center+h*bath.T < hi})
        for a,b in zip(cuts[:-1],cuts[1:]):
            if not b > a:
                raise ArithmeticError('Collapsed baryon panel, unresolved')
            es = a+(xs+1)*(b-a)/2; wt = ws*(b-a)/2
            if np.any(es <= a) or np.any(es >= b) or np.any(wt <= 0):
                raise ArithmeticError('Unrepresentable baryon nodes/weights; no width-only skip')
            energies.extend(es); weights.extend(wt); panels.append((a,b))
    ea = np.repeat(np.array(energies),controls.phi_order)
    wt = np.repeat(np.array(weights),controls.phi_order)/controls.phi_order
    if len(ea) > 250000:
        raise ValueError('Bounded baryon angular array exceeded')
    phi = np.tile(2*math.pi*np.arange(controls.phi_order)/controls.phi_order,len(energies))
    pa = np.sqrt((ea-ma)*(ea+ma))
    ca = ((ma-mb)*(ma+mb)+omega*omega-q*q+2*omega*ea)/(2*pa*q)
    if not np.all(np.isfinite(ca)) or np.any(abs(ca)>1+1e-10):
        raise ArithmeticError('Original delta-shell cosine unresolved')
    ca = np.clip(ca,-1,1); pt = pa*np.sqrt(1-ca*ca)
    a = np.column_stack([ea,pt*np.cos(phi),pt*np.sin(phi),pa*ca])
    b = a+np.array([omega,0,0,q])
    kz = (energy*energy+q*q-pe*pe)/(2*q); kx2 = energy*energy-kz*kz
    if kx2 < -1e-10*energy*energy:
        raise ArithmeticError('Lepton shell unresolved')
    kx = math.sqrt(max(0.,kx2))
    k = np.tile([energy,kx,0,kz],(len(ea),1)); l = np.tile([el,kx,0,kz-q],(len(ea),1))
    pn, pp = (a,b) if channel == 'EC' else (b,a)
    amp = C.scalar_capture(channel,k,pn,l,pp)
    if np.max(abs(C.dot(b,b)/mb**2-1)) > 1e-10 or np.any(b[:,0] < mb):
        raise ArithmeticError('Final positive-energy baryon shell unresolved')
    base = np.log(wt)+np.log(amp)+math.log(pe/(32*math.pi**3*q*energy*C.HBAR))
    signed_mu = bath.mu_e if channel == 'EC' else -bath.mu_e
    lf = log_fd((mub-ea-omega)/bath.T)+log_fd((ea-mua)/bath.T)+float(log_fd((signed_mu-el)/bath.T))
    lb = log_fd((mua-ea)/bath.T)+log_fd((ea+omega-mub)/bath.T)+float(log_fd((el-signed_mu)/bath.T))
    timer.check()
    return SpectralResult(channel,bath,energy,coefficient(float(logsumexp(base+lf))),
                          coefficient(float(logsumexp(base+lb))),controls,
                          ('q-axis exact delta',intervals,tuple(panels),len(ea)),(el,cosine),units='s^-1 MeV^-1 per cosine')


def integrate(channel,bath,energy,controls,timer):
    energy = validate(channel,bath,energy,controls)
    ma,mb = (C.MN,C.MP) if channel == 'EC' else (C.MP,C.MN)
    signed_mu = bath.mu_e if channel == 'EC' else -bath.mu_e
    mid = energy+ma-mb; p0 = math.sqrt(max(0.,mid*mid-C.ME*C.ME))
    width = max(.03*bath.T,(energy+p0)*math.sqrt(bath.T/ma),(energy+p0)**2/(2*ma))
    upper = max(mid,C.ME,signed_mu)+controls.outer_tail*bath.T
    cuts = sorted({C.ME,upper}|{center+h*scale for center in (mid,signed_mu) for scale in (bath.T,width)
                                for h in (-12,-6,-3,-1,-.5,0,.5,1,3,6,12) if C.ME < center+h*scale < upper})
    x,w = leggauss(controls.lepton_order); cs,cw = leggauss(controls.angle_order)
    jf=[]; cb=[]; calls=0; empty=0
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        yl,yh = math.sqrt(lo-C.ME),math.sqrt(hi-C.ME)
        if not yh > yl:
            raise ArithmeticError('Collapsed mapped lepton panel: alternate map or weighted bound required')
        for u,ww in zip(x,w):
            y = yl+(u+1)*(yh-yl)/2; el = C.ME+y*y; weight = ww*(yh-yl)*y
            if not lo < el < hi or not weight > 0:
                raise ArithmeticError('Unrepresentable lepton panel nodes/weights')
            for cosine,v in zip(cs,cw):
                timer.check()
                row = differential(channel,bath,energy,float(el),float(cosine),controls,timer); calls+=1
                if row.j.log is None: empty+=1; continue
                lw = math.log(weight*v)
                jf.append(lw+row.j.log); cb.append(lw+row.true_chi.log)
    if not jf or not cb:
        raise ArithmeticError('Finite-T positive integrated branches unresolved, not empty support')
    timer.check()
    return SpectralResult(channel,bath,energy,coefficient(float(logsumexp(jf))),coefficient(float(logsumexp(cb))),
                          controls,('El=ME+y²',tuple(cuts),'exact q-axis support',calls,empty))
