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
    p.add_argument('--baryon-integral-rtol', type=float, default=1e-6)
    p.add_argument('--conditioning-rtol', type=float, default=1e-6)
    p.add_argument('--density-root-xtol', type=float, default=2e-8)
    p.add_argument('--axis-ratio-root-xtol', type=float, default=2e-7)
    p.add_argument('--onset-margin-atol', type=float, default=1e-5,
                   help='Final central triangle-margin tolerance in fm^-1')
    p.add_argument('--composition-charge-atol', type=float, default=1e-6)
    return p


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _cached_background(path, *, binary_hash, eos_hash, q, energy):
    """Fail closed for partial, corrupted, or mis-keyed cached results."""
    if not path.exists():
        return None
    try:
        provenance = json.loads((path/'run.json').read_text())
        if (provenance['binary_sha256'] != binary_hash
                or provenance['eos']['sha256'] != eos_hash
                or _sha(path/'eos.dat') != eos_hash
                or _sha(path/'rns_rotating.csv') != provenance['rotating']['csv_sha256']
                or _sha(path/'rns_spherical.csv') != provenance['spherical']['csv_sha256']):
            raise ValueError('checksum mismatch')
        command = provenance['command']
        if (command[command.index('-q')+1] != 'tab'
                or not np.isclose(float(command[command.index('-e')+1]), energy, rtol=1e-9, atol=0)
                or not np.isclose(provenance['rotating']['r_ratio'], q, rtol=0, atol=1e-10)):
            raise ValueError('requested parameter mismatch')
        star = load_tabulated_background(path/'rns_rotating.csv',path/'eos.dat')
        if (not np.isclose(star.metadata['r_ratio'],q,rtol=0,atol=1e-10)
                or not np.isclose(star.metadata['e_center'],energy/1e15,rtol=1e-9,atol=0)):
            raise ValueError('native parameter mismatch')
        return star
    except (OSError,ValueError,KeyError,TypeError,IndexError) as exc:
        raise ValueError(f'Invalid/incomplete cache {path}: {exc}; quarantine or remove it before retrying') from exc


def _accept_residual(value, tolerance, label):
    if not np.isfinite(value) or abs(value)>tolerance:
        raise ValueError(f'{label} residual {value} exceeds declared tolerance {tolerance}')


def _channel_accessibility(margin, lepton_fraction):
    """A momentum triangle alone cannot open an absent lepton channel."""
    return (margin>=0) & (lepton_fraction>0)


def _onset_bracket(left, right, channel):
    """Allow a channel to appear inside a sign-changing sampled interval."""
    key = 'D_'+channel+'_fm_inv'
    present = 'central_'+channel+'_present'
    return ((left[present] or right[present])
            and left[key]*right[key]<=0 and left[key]!=right[key])


def run(args):
    tolerance_names = ('baryon_integral_rtol','conditioning_rtol','density_root_xtol',
                       'axis_ratio_root_xtol','onset_margin_atol','composition_charge_atol')
    tolerances = {name:float(getattr(args,name)) for name in tolerance_names}
    if any(not np.isfinite(x) or x<=0 for x in (*tolerances.values(),args.timeout)):
        raise ValueError('Timeout and numerical tolerances must be finite and positive')
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.eos_npz, allow_pickle=False) as a:
        table = EOSTable(**{k:a['table_'+k].copy() for k in EOSTable._fields})
        cn,xp,xe,xm = (a['comp_'+k].copy() for k in ('n','x_p','x_e','x_mu'))
    if (cn.ndim!=1 or any(x.shape != cn.shape for x in (xp,xe,xm)) or len(cn)<4
            or np.any(cn<=0)
            or np.any(np.diff(cn)<=0) or not np.all(np.isfinite(cn))
            or any(not np.all(np.isfinite(x)) or np.any(x<0) or np.any(x>1)
                   for x in (xp,xe,xm))):
        raise ValueError('Require finite increasing composition density and fractions in [0,1]')
    if np.max(np.abs(xp-xe-xm))>args.composition_charge_atol:
        raise ValueError('Composition violates charge neutrality xp=xe+xmu')
    qgrid = np.asarray(args.q_grid)
    if (qgrid.ndim!=1 or qgrid.size==0 or not np.all(np.isfinite(qgrid)) or np.any(np.diff(qgrid)<=0)
            or np.any(qgrid<.6) or np.any(qgrid>1)):
        raise ValueError('q-grid must increase strictly within [.6,1]')
    if (not len(args.initial_nc) or not all(np.isfinite(args.initial_nc)) or min(args.initial_nc)<=0
            or np.shape(args.nc_bracket)!=(2,) or not np.all(np.isfinite(args.nc_bracket))
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
        energy = float(np.exp(np.interp(np.log(nc),np.log(table.n),np.log(table.eps_mevfm3))))
        energy *= MEVFM3_TO_ERGCM3/C_CGS**2
        star = _cached_background(path,binary_hash=binary_hash,eos_hash=eos_manifest['sha256'],
                                  q=q,energy=energy)
        if star is None:
            _,star = solve_tabulated(args.executable,path,table,
                energy_center_g_cm3=energy,axis_ratio=float(q),timeout=args.timeout)
        f,meta = star.fields,star.metadata
        if meta['eos_sha256'] != eos_manifest['sha256']:
            raise ValueError('Cached EOS provenance mismatch')
        n = f['n_fm3']
        central_n = n[0,:]
        if not np.allclose(central_n,central_n[0],rtol=1e-6,atol=0):
            raise ValueError('Central density is angularly inconsistent')
        actual_nc = float(central_n[0])
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
        volume_residual = nb/meta['baryon_number']-1
        _accept_residual(volume_residual,args.baryon_integral_rtol,'Baryon-volume')
        dc_e,dc_m = margins(actual_nc)
        e_access = _channel_accessibility(de,np.interp(n,cn,xe))
        mu_access = _channel_accessibility(dm,np.interp(n,cn,xm))
        comp_domain = (n>=cn[0]) & (n<=cn[-1])
        return dict(requested_nc_fm3=float(nc),nc_fm3=actual_nc,
            requested_axis_ratio=float(q),axis_ratio=meta['r_ratio'],
            frequency_hz=meta['frequency_hz'],Mg=meta['Mass_msun'],
            Mb_native=meta['Mass_b_msun'],Nb=meta['baryon_number'],
            J_g_cm2_s=meta['J_g_cm2_s'],R_eq_km=meta['R_e_km'],
            D_e_fm_inv=float(dc_e),D_mu_fm_inv=float(dc_m),
            central_e_present=bool(np.interp(actual_nc,cn,xe)>0),
            central_mu_present=bool(np.interp(actual_nc,cn,xm)>0),
            Nb_integral_relative_error=volume_residual,
            F_e=integral(weight*e_access*comp_domain)/nb,
            F_mu=integral(weight*mu_access*comp_domain)/nb,
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
                        nc0*args.nc_bracket[0],nc0*args.nc_bracket[1],
                        xtol=args.density_root_xtol,rtol=4*np.finfo(float).eps)
                    result = evaluate(nc,q)
                result.update(condition=condition,initial_nc_fm3=nc0,
                    conditioning_relative_residual=result[condition]/initial[condition]-1)
                _accept_residual(result['conditioning_relative_residual'],
                                 args.conditioning_rtol,'Conditioning')
                return result
            track = [matched(float(q)) for q in qgrid]
            rows.extend(track)
            if args.onsets:
                for channel in ('e','mu'):
                    key = 'D_'+channel+'_fm_inv'
                    for left,right in zip(track[:-1],track[1:]):
                        if _onset_bracket(left,right,channel):
                            q = brentq(lambda q:matched(q)[key],left['requested_axis_ratio'],
                                       right['requested_axis_ratio'],xtol=args.axis_ratio_root_xtol,
                                       rtol=4*np.finfo(float).eps)
                            if any(o['channel']==channel and o['condition']==condition
                                   and o['initial_nc_fm3']==nc0
                                   and abs(o['requested_axis_ratio']-q)<=args.axis_ratio_root_xtol
                                   for o in onsets):
                                continue  # A grid-point zero may bracket both adjacent intervals.
                            onset = matched(q)
                            if not onset['central_'+channel+'_present']:
                                raise ValueError('Onset channel absent at final root')
                            _accept_residual(onset[key],args.onset_margin_atol,'Onset margin')
                            onset.update(channel=channel,margin_residual_fm_inv=onset[key])
                            onsets.append(onset)
            print(json.dumps(dict(initial_nc_fm3=nc0,condition=condition,rows=track)),flush=True)
    result = dict(status='engineering demonstrator; no EOS posterior or cooling evolution',
        initial_matching='identical EOS, central density and first q for both conditions',
        initial_nc_grid=args.initial_nc,q_grid=args.q_grid,conditions=['Nb','Mg'],
        density_bracket_factors=args.nc_bracket,tolerances=tolerances,
        caveats=['Branch stability is not checked; review density bracket and connected branch.',
                 'Finite supplied table surface; no physical crust extrapolation.',
                 'Composition must match EOS; inaccessible/uncovered composition is reported.',
                 'Threshold masks and roots need grid refinement before scientific precision claims.'],
        driver_sha256=_sha(Path(__file__)),
        density_convention='native baryon density at r=0, equator; angular consistency checked',
        eos_npz_sha256=hashlib.sha256(args.eos_npz.read_bytes()).hexdigest(),
        binary_sha256=binary_hash,rows=rows,onsets=onsets)
    (args.output/'tracks.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    run(parser().parse_args())
