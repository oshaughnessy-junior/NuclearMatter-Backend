"""Build and run a pinned external RNS export driver without vendoring RNS.

Based on the public cgca/rns v2 sample interface; see ATTRIBUTION.md.
The supported runner uses polytropic K=G=c=1 units, never Hz or km.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np

RNS_COMMIT = "87ea68246ee990256f94aa91a0506f54d78dd8d0"
MAIN_SHA256 = "5bbe51d50f4f13b7eb3b77fd4564289b7c2b9e8c76523b821ea33c7a3ba78423"
FIELDS = ("rho", "gama", "alpha", "omega", "velocity_sq", "energy", "pressure", "enthalpy")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class RNSBackground:
    """Radial-major RNS export; rho is a metric potential, not mass density."""

    r: np.ndarray
    mu: np.ndarray
    fields: dict
    metadata: dict

    @property
    def s(self):
        return self.r / (self.r + self.metadata["r_e"])


def load_background(path):
    """Reject malformed, nonfinite or convention-inconsistent RNS exports."""
    path = Path(path)
    with path.open() as stream:
        header = stream.readline()
    metadata = {key: float(value) for key, value in re.findall(
        r"([A-Za-z_]+)=([-+0-9.eE]+)", header)}
    if not {"r_ratio", "r_e", "Omega"} <= metadata.keys():
        raise ValueError("Missing RNS coordinate/rotation metadata")
    if (not all(math.isfinite(x) for x in metadata.values())
            or metadata["r_e"] <= 0 or not 0 < metadata["r_ratio"] <= 1):
        raise ValueError("Invalid RNS coordinate/rotation metadata")
    data = np.genfromtxt(path, delimiter=",", names=True, skip_header=1)
    required = {"r", "mu", *FIELDS}
    if data.ndim != 1 or not required <= set(data.dtype.names or ()):
        raise ValueError("Incomplete RNS export schema")
    if not all(np.all(np.isfinite(data[name])) for name in required):
        raise ValueError("Nonfinite RNS export")
    r, mu = np.unique(data["r"]), np.unique(data["mu"])
    if (len(r) < 3 or len(mu) < 3 or r[0] != 0 or mu[0] != 0 or mu[-1] != 1
            or not np.array_equal(data["r"], np.repeat(r, len(mu)))
            or not np.array_equal(data["mu"], np.tile(mu, len(r)))):
        raise ValueError("Expected a complete radial-major equatorial-symmetry grid")
    fields = {name: data[name].reshape(len(r), len(mu)) for name in FIELDS}
    matter = fields["energy"] > 0
    if (not np.any(matter) or np.any(fields["energy"] < 0)
            or np.any(fields["pressure"] < 0)
            or np.any(fields["velocity_sq"][matter] < 0)
            or np.any(fields["velocity_sq"][matter] >= 1)):
        raise ValueError("Invalid RNS matter/velocity support")
    rr, mm = np.meshgrid(r, mu, indexing="ij")
    velocity = ((metadata["Omega"] - fields["omega"]) * rr
                * np.sqrt(1 - mm**2) * np.exp(-fields["rho"]))**2
    error = np.max(np.abs(velocity[matter] - fields["velocity_sq"][matter]))
    if error > 1e-4:
        raise ValueError(f"RNS metric/velocity convention mismatch: {error}")
    if metadata["r_ratio"] == 1 and abs(metadata["Omega"]) > 1e-12:
        raise ValueError("Nonzero rotation in a spherical RNS export")
    if "Omega_K" in metadata and metadata["Omega"] > metadata["Omega_K"]:
        raise ValueError("Requested RNS configuration exceeds mass shedding")
    metadata.update(units="polytropic K=G=c=1", velocity_max_error=float(error),
                    csv_sha256=_sha(path), source_path=str(path.resolve()))
    return RNSBackground(r, mu, fields, metadata)


_EXPORT_TAIL = r'''
  double target_q = 0.8;
  const char *q_env = getenv("NMB_RNS_Q");
  if(q_env) target_q = strtod(q_env, NULL);
  if(!isfinite(target_q) || target_q < 0.6 || target_q > 1) return 3;
  for(int model=0; model<2; model++) {
    if(model==1) {
      int nstep = (int)ceil((1-target_q)/0.025);
      for(int step=1; step<=nstep; step++) {
        r_ratio = 1-(1-target_q)*step/nstep;
        spin(s_gp,mu,log_e_tab,log_p_tab,log_h_tab,log_n0_tab,
          n_tab,eos_type,Gamma_P,h_center,enthalpy_min,rho,gama,alpha,
          omega,energy,pressure,enthalpy,velocity_sq,a_check,accuracy,cf,
          r_ratio,&r_e,&Omega);
      }
    }
    mass_radius(s_gp,mu,log_e_tab,log_p_tab,log_h_tab,log_n0_tab,
      n_tab,eos_type,Gamma_P,rho,gama,alpha,omega,energy,pressure,
      enthalpy,velocity_sq,r_ratio,e_surface,r_e,Omega,
      &Mass,&Mass_0,&J,&R_e,v_plus,v_minus,&Omega_K);
    FILE *out=fopen(model ? "rns_rotating.csv":"rns_spherical.csv","w");
    if(!out) return 2;
    fprintf(out,"# r_ratio=%.17g r_e=%.17g Omega=%.17g h_center=%.17g "
      "e_center=%.17g Gamma=%.17g Mass=%.17g Mass_b=%.17g "
      "J=%.17g R_e=%.17g Omega_K=%.17g\n",
      r_ratio,r_e,Omega,h_center,e_center,Gamma_P,Mass,Mass_0,J,R_e,Omega_K);
    fprintf(out,"r,mu,rho,gama,alpha,omega,velocity_sq,energy,pressure,enthalpy\n");
    for(int si=1;si<SDIV;si++) for(int mi=1;mi<=MDIV;mi++)
      fprintf(out,"%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
        r_e*s_gp[si]/(1-s_gp[si]),mu[mi],rho[si][mi],gama[si][mi],
        alpha[si][mi],omega[si][mi],velocity_sq[si][mi],energy[si][mi],
        pressure[si][mi],enthalpy[si][mi]);
    fclose(out);
  }
  return 0;
}
'''


def export_driver(source, destination):
    """Generate a separate driver from the exact pinned RNS sample main.c."""
    source, destination = Path(source), Path(destination)
    if _sha(source / "main.c") != MAIN_SHA256:
        raise ValueError(f"main.c does not match cgca/rns {RNS_COMMIT}")
    original = (source / "main.c").read_text()
    original = original.replace("n_tab;", "n_tab=0;").replace("a_check;", "a_check=0;")
    # The original parser reads argv[i+1] for every option. Guard that access.
    original = original.replace("switch(argv[i][1]){", "if(i+1>=argc) return 4;\n      switch(argv[i][1]){")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(original[:original.index("  dr=0.05;")] + _EXPORT_TAIL)
    return destination


def build_rns(source, output, *, radial_points=129, angular_points=65, cc="cc"):
    """Compile external RNS on the execution host; record input/binary hashes."""
    if (radial_points < 33 or angular_points < 17
            or (radial_points - 1) % 2 or (angular_points - 1) % 2):
        raise ValueError("RNS Simpson grids need odd sizes >=33 radial, >=17 angular")
    source, output = Path(source).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    driver = export_driver(source, output / "thermal_export.c")
    executable = output / "rns_export"
    command = [cc, "-std=c99", "-O2", f"-DMDIV={angular_points}",
               f"-DSDIV={radial_points}", "-I", str(source), "-o", str(executable),
               str(driver), *(str(source / name) for name in
                              ("equil.c", "equil_util.c", "nrutil.c")), "-lm"]
    result = subprocess.run(command, text=True, capture_output=True, timeout=120)
    (output / "build.log").write_text(result.stdout + result.stderr)
    result.check_returncode()
    inputs = {path.name: _sha(path) for path in source.iterdir()
              if path.suffix in (".c", ".h") and path.name != "thermal_export.c"}
    (output / "build.json").write_text(json.dumps({
        "upstream_commit": RNS_COMMIT, "source_sha256": inputs,
        "generated_driver_sha256": _sha(driver), "command": command,
        "binary_sha256": _sha(executable), "radial_points": radial_points,
        "angular_points": angular_points,
    }, indent=2) + "\n")
    return executable


def solve_polytrope(executable, output, *, energy_center=0.2, index=1.0,
                    axis_ratio=0.8, timeout=120):
    """Run conventional RNS in a fresh directory; reject unphysical exports.

    The central value is energy density, not rest-mass density. The external
    process is not an AD primitive. Passing a float derived from JAX does not
    differentiate the equilibrium solve.
    """
    if (not all(math.isfinite(x) for x in (energy_center, index, axis_ratio))
            or energy_center <= 0 or index <= 0 or not 0.6 <= axis_ratio <= 1):
        raise ValueError("Require positive energy/index and 0.6 <= axis ratio <= 1")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    executable = Path(executable).resolve()
    command = [str(executable), "-q", "poly", "-N", str(index), "-e", str(energy_center)]
    started = time.monotonic()
    result = subprocess.run(command, cwd=output, capture_output=True, text=True,
                            timeout=timeout, env={**os.environ, "NMB_RNS_Q": str(axis_ratio)})
    (output / "run.log").write_text(result.stdout + result.stderr)
    result.check_returncode()
    spherical = load_background(output / "rns_spherical.csv")
    rotating = load_background(output / "rns_rotating.csv")
    if abs(rotating.metadata["r_ratio"] - axis_ratio) > 1e-12:
        raise ValueError("RNS export did not honor requested axis ratio")
    (output / "run.json").write_text(json.dumps({
        "command": command, "axis_ratio": axis_ratio,
        "binary_sha256": _sha(executable), "elapsed_seconds": time.monotonic() - started,
        "spherical": spherical.metadata, "rotating": rotating.metadata,
    }, indent=2) + "\n")
    return spherical, rotating
