"""Reproduce bounded Newtonian rotating-mode and eigenproblem checks.

An optional RNS file exercises input exchange only. No relativistic mode
frequency is manufactured from a polytrope export or a fitted relation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
from nmbackend.modes import solve,toroidal_rmode,bar_mode,neutrality_thresholds,rns_polytropic_background
from nmbackend.rotating.rns import load_background


def complex_rows(values):return [{'real':float(x.real),'imag':float(x.imag)} for x in values]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('output',type=Path)
    p.add_argument('--rns-csv',type=Path)
    args=p.parse_args()
    result={'scope':'numerical mode infrastructure and Newtonian physical benchmarks; no rotating full-GR action assembly',
            'python':sys.version,'jax':jax.__version__,'host':platform.node(),'harmonic':'exp(i*(sigma*t+m*phi))',
            'r_modes':[],'maclaurin':[],'relativistic_rotating_modes_computed':False}
    for m in (2,3,4):
        for order in (2*m+2,24,40):
            pencil,meta=toroidal_rmode(m,1.,angular_points=order)
            modes=solve(pencil)
            frequency=modes.sigma[0].real+m
            result['r_modes'].append({'m':m,'angular_points':order,'omega_ro_over_Omega':float(frequency),
                'analytic_error':float(abs(frequency-2/(m+1))),'residual':float(modes.residual[0])})
    for e in (0.,.2,.5,.8,.9,.95,.97):
        pencil,meta=bar_mode(e)
        modes=solve(pencil)
        result['maclaurin'].append({**meta,'sigma':complex_rows(modes.sigma),'max_residual':float(max(modes.residual))})
    result['maclaurin_thresholds']=neutrality_thresholds()
    if args.rns_csv:
        star=load_background(args.rns_csv)
        response=rns_polytropic_background(star)
        matter=star.fields['energy']>0
        result['rns_input_contract']={'csv_sha256':star.metadata['csv_sha256'],'axis_ratio':star.metadata['r_ratio'],
            'closure':response.closure,'sound_speed_sq_min':float(np.min(response.sound_speed_sq[matter])),
            'sound_speed_sq_max':float(np.max(response.sound_speed_sq[matter])),
            'scope':'exchange contract only; full-GR matrices and spectral background qualification remain open'}
    root=Path(__file__).resolve().parents[1]/'src/nmbackend/modes'
    result['source_sha256']={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in root.glob('*.py')}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'thresholds':result['maclaurin_thresholds'],'r_mode_max_error':max(x['analytic_error'] for x in result['r_modes']),
        'maclaurin_max_residual':max(x['max_residual'] for x in result['maclaurin']),'rns_input_contract':result.get('rns_input_contract')},indent=2))


if __name__=='__main__':main()
