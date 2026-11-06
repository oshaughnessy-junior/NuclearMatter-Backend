"""Check independently implemented KEH equations on conventional RNS fields."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import RectBivariateSpline
from nmbackend.rotating.rns import load_background
from nmbackend.rotating.pino.keh import Fields, make_grid, alpha_from_fields, source_terms, green_map, matter_from_fields, residual_and_observables
from nmbackend.rotating.pino.polytrope import eos_at_enthalpy


def reference_fields(star, grid, stride=1):
    names = ('gama', 'rho', 'omega')
    # The known asymptotic limit is zero at compactified infinity.
    s = np.append(star.s[::stride], 1.)
    splines = [RectBivariateSpline(s, star.mu, np.vstack([star.fields[n][::stride], np.zeros(len(star.mu))]), kx=3, ky=3) for n in names]
    ss, mm = np.meshgrid(grid.s, grid.mu, indexing='ij')
    def values(s, mu, dx=0, dy=0):
        return np.stack([spline.ev(s, mu, dx=dx, dy=dy) for spline in splines], axis=-1)
    arrays = [values(ss, mm, dx, dy) for dx, dy in ((0,0),(1,0),(0,1),(2,0),(1,1),(0,2))]
    meta = star.metadata
    q = meta['r_ratio']
    return Fields(*(jnp.asarray(a) for a in arrays),
        jnp.asarray(values(np.asarray(grid.s), np.ones(len(grid.s)))),
        jnp.asarray(values(0.,0.)), jnp.asarray(values(q/(1+q),1.)),
        jnp.asarray(values(.5,0.)), jnp.asarray(meta['r_e']),
        jnp.asarray(meta['Omega']), jnp.asarray(meta['h_center']), jnp.asarray(0.))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv', type=Path)
    ap.add_argument('output', type=Path)
    ap.add_argument('--diagnostic-only', action='store_true', help='Preserve diagnostics without declaring a gate pass')
    ap.add_argument('--radial-stride',type=int,default=1)
    args = ap.parse_args()
    star = load_background(args.csv)
    index = 1/(star.metadata['Gamma']-1)
    records = []
    for nr in (32,64,128):
        grid = make_grid(nr, 16, 6)
        fields = reference_fields(star, grid,args.radial_stride)
        alpha = alpha_from_fields(fields, grid)
        ss, mm = np.meshgrid(grid.s, grid.mu, indexing='ij')
        alpha_spline = RectBivariateSpline(np.append(star.s,1.),star.mu,np.vstack([star.fields['alpha'], np.zeros(len(star.mu))]))
        exact_alpha = alpha_spline.ev(ss,mm)
        eos = lambda h: eos_at_enthalpy(h,index)
        loss, diag = residual_and_observables(fields,grid,eos)
        matter = matter_from_fields(fields,grid,eos)
        mapped = green_map(source_terms(fields,jnp.asarray(exact_alpha),matter,grid),fields,grid)
        absolute = np.max(np.abs(np.asarray(mapped-fields.value)),axis=(0,1))
        relative = np.sqrt(np.mean(np.asarray(mapped-fields.value)**2,axis=(0,1))/np.maximum(np.mean(np.asarray(fields.value)**2,axis=(0,1)),1e-16))
        record = {k:np.asarray(v).tolist() for k,v in diag.items()}
        location=np.unravel_index(np.argmax(np.abs(alpha-exact_alpha)),alpha.shape)
        sm=np.asarray(grid.s)*(1-np.asarray(grid.s))
        dg=sm*np.asarray(fields.ds[:,0,0])
        d2g=sm*((1-2*np.asarray(grid.s))*np.asarray(fields.ds[:,0,0])+sm*np.asarray(fields.dss[:,0,0]))
        dnu=sm*np.asarray(fields.ds[:,0,0]+fields.ds[:,0,1])/2
        spherical_identity=.5*(d2g+dg*dg)+dnu*dnu-(1+dg)*dg
        record.update(radial_stride=args.radial_stride,alpha_error_location=[float(grid.s[location[0]]),float(grid.mu[location[1]])],spherical_identity_max=float(np.max(np.abs(spherical_identity))),spherical_identity_at_alpha_error=float(spherical_identity[location[0]]))
        record.update(nr=nr,nmu=16,harmonics=6,alpha_absolute_error=float(np.max(np.abs(alpha-exact_alpha))), green_absolute_error=absolute.tolist(),green_rms_relative_error=relative.tolist(),mass_relative_error=float(diag['mass']/star.metadata['Mass']-1),radius_relative_error=float(diag['radius']/star.metadata['R_e']-1))
        records.append(record)
        print(json.dumps(record),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'reference':star.metadata,'records':records},indent=2)+'\n')
    if args.diagnostic_only:
        return
    # A development gate, not a published inference accuracy target.
    final=records[-1]
    assert final['alpha_absolute_error'] < .01
    assert max(final['green_rms_relative_error'][:2]) < .05
    if star.metadata['Omega']>0:
        assert final['green_rms_relative_error'][2] < .05
    assert abs(final['mass_relative_error']) < .02
    assert abs(final['radius_relative_error']) < .002


if __name__=='__main__':
    main()
