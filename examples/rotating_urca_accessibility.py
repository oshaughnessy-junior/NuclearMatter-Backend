"""Engineering-only rotating Urca-accessibility tracks; no EOS posterior/cooling.

Input NPZ contains table_<EOSTable field> plus comp_n, comp_x_p, comp_x_e,
comp_x_mu. Composition must belong to the same EOS; this script cannot verify
that scientific assumption. The chosen density brackets require branch review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import simpson
from scipy.optimize import brentq

from nmbackend.eos.base import EOSTable
from nmbackend.rotating.tabulated import (
    C_CGS, MEVFM3_TO_ERGCM3, export_eos, load_tabulated_background, solve_tabulated,
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eos-npz', type=Path, required=True)
    p.add_argument('--executable', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--initial-nc', type=float, nargs='+', required=True,
                   help='Engineering initial baryon densities in fm^-3')
    p.add_argument('--q-grid', type=float, nargs='+', default=[.8,.85,.9,.95,1.])
    p.add_argument('--nc-bracket', type=float, nargs=2, default=[.8,1.7],
                   help='Root-bracket factors times each initial density')
    p.add_argument('--onsets', action='store_true',
                   help='Additionally root-find central-margin onsets within sampled q brackets')
    p.add_argument('--timeout', type=float, default=120)
    return p


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.eos_npz, allow_pickle=False) as a:
        table = EOSTable(**{k:a['table_'+k].copy() for k in EOSTable._fields})
        cn,xp,xe,xm = (a['comp_'+k].copy() for k in ('n','x_p','x_e','x_mu'))
    if (any(x.shape != cn.shape for x in (xp,xe,xm)) or len(cn)<4
            or np.any(np.diff(cn)<=0) or not np.all(np.isfinite(cn))
            or any(not np.all(np.isfinite(x)) or np.any(x<0) or np.any(x>1)
                   for x in (xp,xe,xm))):
        raise ValueError('Require finite increasing composition density and fractions in [0,1]')
    qgrid = np.asarray(args.q_grid)
    if (not np.all(np.isfinite(qgrid)) or np.any(np.diff(qgrid)<=0)
            or np.any(qgrid<.6) or np.any(qgrid>1)):
        raise ValueError('q-grid must increase strictly within [.6,1]')
    if (not all(np.isfinite(args.initial_nc)) or min(args.initial_nc)<=0
            or not 0<args.nc_bracket[0]<1<args.nc_bracket[1]):
        raise ValueError('Require positive initial densities and bracket factors enclosing 1')
    digest = hashlib.sha256(b''.join(np.asarray(x).tobytes() for x in table)).hexdigest()
    reference = args.output/f'eos-{digest[:16]}.dat'
    eos_manifest = export_eos(table,reference)
    binary_hash = hashlib.sha256(args.executable.read_bytes()).hexdigest()
    cache_root = args.output/'runs'/f'{eos_manifest["sha256"][:16]}-{binary_hash[:16]}'

    def margins(n):
        k = (3*np.pi**2*np.maximum(n,0))**(1/3)
        p,e,m = (np.interp(n,cn,x) for x in (xp,xe,xm))
        return k*(p**(1/3)+e**(1/3)-(1-p)**(1/3)), k*(p**(1/3)+m**(1/3)-(1-p)**(1/3))

    def evaluate(nc,q):
        if not float(table.n[0])<nc<float(table.n[-1]):
            raise ValueError('Requested central density outside EOS table')
        # Preserve q resolution for nested onset roots; .6f would quantize roots.
        path = cache_root/f'n{nc:.10f}-q{q:.10f}'
        if (path/'run.json').exists():
            provenance = json.loads((path/'run.json').read_text())
            if provenance['binary_sha256'] != binary_hash:
                raise ValueError('Cached executable provenance mismatch')
            star = load_tabulated_background(path/'rns_rotating.csv',path/'eos.dat')
        else:
            energy = np.exp(np.interp(np.log(nc),np.log(table.n),np.log(table.eps_mevfm3)))
            _,star = solve_tabulated(args.executable,path,table,
                energy_center_g_cm3=float(energy)*MEVFM3_TO_ERGCM3/C_CGS**2,
                axis_ratio=float(q),timeout=args.timeout)
        f,meta = star.fields,star.metadata
        if meta['eos_sha256'] != eos_manifest['sha256']:
            raise ValueError('Cached EOS provenance mismatch')
        n = f['n_fm3']
        actual_nc = float(n.max())
        if not cn[0]<=actual_nc<=cn[-1]:
            raise ValueError('Central density outside supplied composition domain')
        de,dm = margins(n)
        s = star.s
        boundary = eos_manifest['energy_surface_g_cm3']/1e15
        matter = f['energy']>boundary
        weight = np.where(matter,n/np.sqrt(1-f['velocity_sq'])*
            np.exp(2*f['alpha']+.5*(f['gama']-f['rho'])),0)
        weight *= (star.r[:,None]**2)*(meta['r_e']/(1-s[:,None])**2)

        def integral(w):
            full_s = np.r_[s,s[-1]+s[1]-s[0]]
            full = np.concatenate((w,np.zeros((1,len(star.mu)))),axis=0)
            return float(4*np.pi*simpson(simpson(full,x=star.mu,axis=1),x=full_s)
                         *meta['length_scale_cm']**3*1e39)

        nb = integral(weight)
        dc_e,dc_m = margins(actual_nc)
        comp_domain = (n>=cn[0]) & (n<=cn[-1])
        return dict(requested_nc_fm3=float(nc),nc_fm3=actual_nc,
            requested_axis_ratio=float(q),axis_ratio=meta['r_ratio'],
            frequency_hz=meta['frequency_hz'],Mg=meta['Mass_msun'],
            Mb_native=meta['Mass_b_msun'],Nb=meta['baryon_number'],
            J_g_cm2_s=meta['J_g_cm2_s'],R_eq_km=meta['R_e_km'],
            D_e_fm_inv=float(dc_e),D_mu_fm_inv=float(dc_m),
            Nb_integral_relative_error=nb/meta['baryon_number']-1,
            F_e=integral(weight*(de>=0)*comp_domain)/nb,
            F_mu=integral(weight*(dm>=0)*comp_domain)/nb,
            composition_coverage=integral(weight*comp_domain)/nb,
            source=str(path.resolve()),eos_sha256=meta['eos_sha256'])

    rows,onsets = [],[]
    for nc0 in args.initial_nc:
        initial = evaluate(nc0,qgrid[0])
        for condition in ('Nb','Mg'):
            def matched(q):
                if q==qgrid[0]:
                    result = dict(initial)
                else:
                    target = initial[condition]
                    nc = brentq(lambda nc:evaluate(nc,q)[condition]/target-1,
                        nc0*args.nc_bracket[0],nc0*args.nc_bracket[1],xtol=2e-7,rtol=2e-7)
                    result = evaluate(nc,q)
                result.update(condition=condition,initial_nc_fm3=nc0,
                    conditioning_relative_residual=result[condition]/initial[condition]-1)
                return result
            track = [matched(float(q)) for q in qgrid]
            rows.extend(track)
            if args.onsets:
                for channel in ('e','mu'):
                    key = 'D_'+channel+'_fm_inv'
                    for left,right in zip(track[:-1],track[1:]):
                        if left[key]*right[key]<=0 and left[key]!=right[key]:
                            q = brentq(lambda q:matched(q)[key],left['requested_axis_ratio'],
                                       right['requested_axis_ratio'],xtol=2e-8,rtol=2e-8)
                            onset = matched(q)
                            onset.update(channel=channel)
                            onsets.append(onset)
            print(json.dumps(dict(initial_nc_fm3=nc0,condition=condition,rows=track)),flush=True)
    result = dict(status='engineering demonstrator; no EOS posterior or cooling evolution',
        initial_matching='identical EOS, central density and first q for both conditions',
        initial_nc_grid=args.initial_nc,q_grid=args.q_grid,conditions=['Nb','Mg'],
        density_bracket_factors=args.nc_bracket,
        caveats=['Branch stability is not checked; review density bracket and connected branch.',
                 'Finite supplied table surface; no physical crust extrapolation.',
                 'Composition must match EOS; inaccessible/uncovered composition is reported.',
                 'Threshold masks and roots need grid refinement before scientific precision claims.'],
        eos_npz_sha256=hashlib.sha256(args.eos_npz.read_bytes()).hexdigest(),
        binary_sha256=binary_hash,rows=rows,onsets=onsets)
    (args.output/'tracks.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    run(parser().parse_args())
