"""Physical tabulated-EOS exchange with pinned external RNS (not differentiable).

RNS uses legacy CGS constants and scaled geometry internally. Keep these units
explicit: its global Omega is rad/s, whereas exported frame dragging is scaled.
The finite low-density EOSTable boundary is matched, without extrapolation.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

import numpy as np

from ..constants import MSUN_SI
from .rns import RNS_COMMIT, RNSBackground, _sha, export_driver, load_background

C_CGS = 2.9979e10
G_CGS = 6.6732e-8
MB_G = 1.66e-24
LENGTH_CM = math.sqrt(1.e-15 * C_CGS**2 / G_CGS)
MEVFM3_TO_ERGCM3 = 1.602176634e33
MAX_ROWS = 8192


def export_eos(table, path):
    """Write native rho,p,c²h,n table, with units and boundary provenance."""
    path = Path(path)
    n, eps, p, h = (np.asarray(getattr(table, key), dtype=float)
                    for key in ('n', 'eps_mevfm3', 'p_mevfm3', 'h'))
    if (not 4 <= len(n) <= MAX_ROWS or any(a.shape != n.shape for a in (eps,p,h))
            or any(not np.all(np.isfinite(a)) or np.any(a <= 0)
                   or np.any(np.diff(a) <= 0) for a in (n,eps,p,h))):
        raise ValueError('RNS requires 4..8192 finite positive strictly increasing EOS rows')
    rows = np.column_stack((eps*MEVFM3_TO_ERGCM3/C_CGS**2,
                            p*MEVFM3_TO_ERGCM3, h*C_CGS**2, n*1.e39))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, rows, header=str(len(n)), comments='', fmt='%.17e')
    manifest = dict(format='RNS rho,p,c2h,n0', columns=['g/cm^3','dyne/cm^2','erg/g','cm^-3'],
                    rows=len(n), sha256=_sha(path), surface='finite EOSTable minimum',
                    c_cgs=C_CGS, g_cgs=G_CGS, baryon_mass_g=MB_G,
                    enthalpy_min=float(h[0]), energy_surface_g_cm3=float(rows[0,0]),
                    pressure_surface_dyn_cm2=float(rows[0,1]))
    path.with_suffix(path.suffix+'.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def build_tabulated_rns(source, output, *, radial_points=129, angular_points=65, cc='cc'):
    """Build a derived driver with full table capacity and matching surface."""
    if radial_points < 33 or angular_points < 17 or radial_points%2 != 1 or angular_points%2 != 1:
        raise ValueError('Require odd RNS Simpson grids >=33 radial, >=17 angular')
    source, output = Path(source).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    driver = export_driver(source, output/'tabulated_export.c')
    text = driver.read_text().replace('[201]', f'[{MAX_ROWS+1}]')
    for old, new in (
        ('e_surface=7.8*C*C*KSCALE;', 'e_surface=strtod(getenv("NMB_TAB_ES"),NULL)*C*C*KSCALE;'),
        ('p_surface=1.01e8*KSCALE;', 'p_surface=strtod(getenv("NMB_TAB_PS"),NULL)*KSCALE;'),
        ('enthalpy_min=1.0/(C*C);', 'enthalpy_min=strtod(getenv("NMB_TAB_HS"),NULL);')):
        if old not in text:
            raise ValueError('Pinned driver surface contract changed')
        text = text.replace(old,new)
    # Fail cleanly if somebody invokes the derived driver outside this runner.
    text = text.replace('e_surface=strtod',
        'if(!getenv("NMB_TAB_ES") || !getenv("NMB_TAB_PS") || !getenv("NMB_TAB_HS")) return 8;\n    e_surface=strtod')
    driver.write_text(text)
    executable = output/'rns_tabulated'
    command = [cc,'-std=c99','-O2',f'-DMDIV={angular_points}',f'-DSDIV={radial_points}',
               '-I',str(source),'-o',str(executable),str(driver),
               *(str(source/name) for name in ('equil.c','equil_util.c','nrutil.c')),'-lm']
    result = subprocess.run(command,capture_output=True,text=True,timeout=120)
    (output/'build.log').write_text(result.stdout+result.stderr)
    result.check_returncode()
    (output/'build.json').write_text(json.dumps(dict(upstream_commit=RNS_COMMIT,
        source_sha256={p.name:_sha(p) for p in source.iterdir() if p.suffix in ('.c','.h')},
        generated_driver_sha256=_sha(driver), binary_sha256=_sha(executable),command=command,
        radial_points=radial_points,angular_points=angular_points,max_eos_rows=MAX_ROWS,
        c_cgs=C_CGS,g_cgs=G_CGS,baryon_mass_g=MB_G),indent=2)+'\n')
    return executable


def _native_log_interp(x, xp, yp):
    """Mirrors RNS interp's local four-point polynomial, on logarithms."""
    x = np.asarray(x)
    start = np.clip(np.searchsorted(xp,x,side='right')-2,0,len(xp)-4)
    xx = xp[start[...,None]+np.arange(4)]
    yy = yp[start[...,None]+np.arange(4)]
    result = np.zeros_like(x,dtype=float)
    for j in range(4):
        term = yy[...,j].copy()
        for k in range(4):
            if k != j:
                term *= (x-xx[...,k])/(xx[...,j]-xx[...,k])
        result += term
    return result


def load_tabulated_background(path, eos_path):
    """Validate scaled geometry; attach physical units and native n(e) bridge.

    Geometry arrays retain the RNS length scale LENGTH_CM. metadata Omega is
    converted to its matching internal unit; Omega_rad_s retains physical spin.
    Mass/Radius physical convenience values have explicit unit suffixes.
    """
    path, eos_path = Path(path), Path(eos_path)
    original = path.read_text()
    raw = {k:float(v) for k,v in re.findall(r'([A-Za-z_]+)=([-+0-9.eE]+)',original.splitlines()[0])}
    internal_omega = raw['Omega']*LENGTH_CM/C_CGS
    header = original.splitlines()[0]
    header = re.sub(r'(?<![A-Za-z_])Omega=[-+0-9.eE]+',f'Omega={internal_omega:.17g}',header)
    if 'Omega_K' in raw:
        header = re.sub(r'Omega_K=[-+0-9.eE]+',f'Omega_K={raw["Omega_K"]*LENGTH_CM/C_CGS:.17g}',header)
    with tempfile.TemporaryDirectory() as temporary:
        normalized = Path(temporary)/'normalized.csv'
        normalized.write_text(header+'\n'+'\n'.join(original.splitlines()[1:])+'\n')
        star = load_background(normalized)
    rows = np.loadtxt(eos_path,skiprows=1)
    manifest = json.loads(eos_path.with_suffix(eos_path.suffix+'.json').read_text())
    if _sha(eos_path) != manifest['sha256']:
        raise ValueError('EOS provenance checksum mismatch')
    fields = dict(star.fields)
    matter = fields['energy'] > 0
    n = np.zeros_like(fields['energy'])
    loge = np.log10(rows[:,0]/1.e15)
    sampled = np.log10(fields['energy'][matter])
    if sampled.size and (sampled.min() < loge[0]-1.e-5 or sampled.max() > loge[-1]+1.e-5):
        raise ValueError('RNS matter outside exported EOS range')
    n[matter] = 10**_native_log_interp(sampled,loge,np.log10(rows[:,3]))/1.e39
    fields['n_fm3'] = n
    metadata = dict(star.metadata)
    metadata.pop('Gamma',None)  # native default Gamma is irrelevant for tabulated EOS
    metadata.update(units='RNS scaled tabulated geometry; explicit CGS conversion metadata',
        eos_kind='tabulated',Omega_rad_s=raw['Omega'],frequency_hz=raw['Omega']/(2*np.pi),
        length_scale_cm=LENGTH_CM,c_cgs=C_CGS,g_cgs=G_CGS,baryon_mass_g=MB_G,
        source_path=str(path.resolve()),csv_sha256=_sha(path),eos_sha256=_sha(eos_path),
        native_header=raw,surface=manifest['surface'])
    for key in ('Mass','Mass_b'):
        if key in raw:
            metadata[key+'_msun'] = raw[key]/(MSUN_SI*1000)
    if 'R_e' in raw:
        metadata['R_e_km'] = raw['R_e']/1.e5
    if 'J' in raw:
        metadata['J_g_cm2_s'] = raw['J']
    if 'Mass_b' in raw:
        metadata['baryon_number'] = raw['Mass_b']/MB_G
    return RNSBackground(star.r,star.mu,fields,metadata)


def solve_tabulated(executable, output, table, *, energy_center_g_cm3, axis_ratio=.8, timeout=120):
    """Solve one central total energy/c² and axis ratio, plus spherical control."""
    if not math.isfinite(energy_center_g_cm3) or energy_center_g_cm3 <= 0 or not .6 <= axis_ratio <= 1:
        raise ValueError('Require positive total energy/c² and .6<=axis_ratio<=1')
    executable,output = Path(executable).resolve(),Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    eos_path = output/'eos.dat'
    manifest = export_eos(table,eos_path)
    rows = np.loadtxt(eos_path,skiprows=1)
    if not rows[0,0] < energy_center_g_cm3 < rows[-1,0]:
        raise ValueError('Central energy must lie strictly inside EOS table')
    command = [str(executable),'-q','tab','-f','eos.dat','-e',str(energy_center_g_cm3)]
    environment = {**os.environ,'NMB_RNS_Q':str(axis_ratio),
        'NMB_TAB_ES':str(manifest['energy_surface_g_cm3']),
        'NMB_TAB_PS':str(manifest['pressure_surface_dyn_cm2']),
        'NMB_TAB_HS':str(manifest['enthalpy_min'])}
    started = time.monotonic()
    result = subprocess.run(command,cwd=output,capture_output=True,text=True,env=environment,timeout=timeout)
    (output/'run.log').write_text(result.stdout+result.stderr)
    result.check_returncode()
    spherical = load_tabulated_background(output/'rns_spherical.csv',eos_path)
    rotating = load_tabulated_background(output/'rns_rotating.csv',eos_path)
    if abs(rotating.metadata['r_ratio']-axis_ratio)>1.e-12:
        raise ValueError('RNS did not honor requested axis ratio')
    (output/'run.json').write_text(json.dumps(dict(command=command,binary_sha256=_sha(executable),
        elapsed_seconds=time.monotonic()-started,eos=manifest,spherical=spherical.metadata,
        rotating=rotating.metadata),indent=2)+'\n')
    return spherical,rotating
