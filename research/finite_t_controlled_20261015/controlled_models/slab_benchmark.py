#!/usr/bin/env python3
"""Exact static occupation-number slab oracle; no nuclear microphysics supplied.

Units choose slab length L=1 and c=1; all opacities are inverse length.
Emissivity j and true absorption chi obey df/ds = j-(j+chi)f.
This is a convention/conservation test, not an NMB rate validation.
"""
from __future__ import annotations
import json
import math
from pathlib import Path

ANGLES = ((0.21132486540518713, 0.5), (0.7886751345948129, 0.5))
ENERGIES = (0.5, 2.0, 8.0, 24.0)


def occupation(distance: float, mu: float, incoming: float, equilibrium: float, opacity: float) -> float:
    return incoming + (equilibrium-incoming) * (-math.expm1(-opacity*distance/mu))


def source_integral(n: int, mu: float, incoming: float, eq: float, opacity: float) -> float:
    # Independent trapezoidal integration over x of C/c=opacity*(eq-f).
    result = 0.0
    for index in range(n + 1):
        factor = 0.5 if index in (0, n) else 1.0
        result += factor * opacity * (eq - occupation(index/n, mu, incoming, eq, opacity))
    return result/n


def run() -> dict:
    worst_conservation = 0.0
    worst_derivative = 0.0
    checks = 0
    for opacity in (0.0, 1.0e-8, 0.1, 1.0, 100.0):
        for energy in ENERGIES:
            eq = 1.0/(math.exp((energy-1.0)/3.0)+1.0)
            # j/(j+chi)=eq; true chi alone is NOT attenuation.
            j, chi = opacity*eq, opacity*(1.0-eq)
            assert abs(j*(1.0-eq)-chi*eq) < 1.0e-14
            source_by_resolution = []
            boundary_number = 0.0
            boundary_energy = 0.0
            base_n = 128 * max(1, math.ceil(opacity))
            for n in (base_n, 2*base_n):
                integral_number = 0.0
                integral_energy = 0.0
                for mu, weight in ANGLES:
                    for incoming in (0.05, 0.8):
                        outgoing = occupation(1.0, mu, incoming, eq, opacity)
                        assert 0.0 <= outgoing <= 1.0
                        boundary = weight*mu*(outgoing-incoming)
                        integral = weight*source_integral(n, mu, incoming, eq, opacity)
                        integral_number += integral
                        integral_energy += energy*integral
                        if n == base_n:
                            boundary_number += boundary
                            boundary_energy += energy*boundary
                        if opacity == 0.0:
                            assert outgoing == incoming
                        if opacity == 100.0:
                            assert abs(outgoing-eq) < 1.0e-14
                        checks += 1
                # Energy is an independent moment; one physical energy group here.
                scale = max(abs(boundary_number), 1.0e-12)
                err = abs(integral_number-boundary_number)/scale
                source_by_resolution.append(err)
                if opacity <= 1.0:
                    assert err < 1.0e-4
                    worst_conservation = max(worst_conservation, err)
                assert abs(integral_energy-energy*integral_number) < 1.0e-12
                assert abs(boundary_energy-energy*boundary_number) < 1.0e-12
            if opacity >= 0.1 and source_by_resolution[0] > 1.0e-7:
                assert source_by_resolution[1] < 0.27*source_by_resolution[0], (opacity, energy, source_by_resolution)
    for mu, _ in ANGLES:
        for opacity in (0.1, 1.0, 3.0):
            incoming, eq, h = 0.1, 0.7, 1.0e-5
            exact = -(incoming-eq)*math.exp(-opacity/mu)/mu
            fd = (occupation(1.0, mu, incoming, eq, opacity+h)-occupation(1.0, mu, incoming, eq, opacity-h))/(2.0*h)
            error = abs(fd-exact)/max(abs(exact), 1.0e-10)
            worst_derivative = max(worst_derivative, error)
            assert error < 1.0e-5
    return {"status": "PASS", "characteristic_checks": checks,
            "max_moderate_depth_conservation_relative_error": worst_conservation,
            "max_opacity_derivative_relative_error": worst_derivative,
            "scope": "Exact static homogeneous emission/absorption occupation slab; no scattering or physical rates validated",
            "conventions": "c=L=1; attenuation=j+chi; angular weights integrate each hemisphere"}


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
    Path(__file__).with_name("slab_benchmark_result.json").write_text(json.dumps(result, indent=2)+"\n")
