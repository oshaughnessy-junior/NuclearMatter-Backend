"""Production evidence_fn: per-EOS evidence from a RIFT CIP run (tabular method).

Glue between the adaptive sequence-space prior (:mod:`nmbackend.priors.gp_pca`)
and RIFT's tabular per-EOS evidence.  One CIP run per refinement round:

1. write the round's draws as a ``pca_hc/1`` file (:func:`write_pca_draws`) --
   CIP opens it through ``EOSManager.EOSSequenceFromFile``;
2. run CIP with ``--tabular-eos-file <draws> --input-eos-index`` against the
   trained likelihood interpolant (``all.net``), requesting posterior samples;
3. read the posterior samples (ligolw XML; the EOS index is serialized in the
   sim_inspiral ``alpha`` column via ``lalsimutils``) and convert the per-EOS
   **posterior mass** into per-EOS log-evidence: under the uniform-in-sequence
   prior CIP samples, ``p(e|d) \\propto Z_e``, so ``lnZ_e = ln(mass_e) + const``
   -- exactly the weights :meth:`CoefficientPrior.refine` consumes (the constant
   cancels).

Cardinality note (IMPORTANT): an ``all.net`` built by a previous ILE/CIP cycle
carries per-point EOS indices bound to the sequence it was produced with.  For a
*new* draw file the index column must reference valid indices; ``remap_index_mod``
rewrites it as ``index % n_draws`` per round.  That is the correct *plumbing/demo*
configuration (same caveat as the live smoke test); a production paper-1 run
regenerates ``all.net`` with conventions matched to the draw cardinality.

This module shells out to CIP (subprocess) and lazily imports ``RIFT.lalsimutils``
only to read the samples -- it requires a working RIFT+LAL environment at call
time, not at import time.
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

from ..priors.gp_pca import write_pca_draws

#: CIP argument template for the GW170817-style tabular run; {net} and {seq} are
#: filled per round.  Override via ``extra_args`` / constructor knobs as needed.
DEFAULT_CIP_ARGS = (
    "--fname {net} --parameter mc --parameter delta_mc --parameter-implied xi "
    "--parameter-implied chiMinus --parameter-nofit s1z --parameter-nofit s2z "
    "--input-tides --input-eos-index --tabular-eos-file {seq} "
    "--lambda-max 20000 --fit-method rf --no-plots "
    "--mc-range [1.197,1.1985] --eta-range [0.2,0.249999] "
    "--downselect-parameter m1 --downselect-parameter-range [1,3] "
    "--downselect-parameter m2 --downselect-parameter-range [1,3] "
    "--sampler-method default --aligned-prior alignedspin_zprior --chi-max 0.05"
)


def posterior_mass_lnZ(eos_indices, n_eos, floor_counts=0.1):
    """Per-EOS log-evidence (up to a constant) from posterior-sample EOS indices.

    ``mass_e \\propto Z_e`` under a uniform sequence prior; Laplace-style
    ``floor_counts`` keeps unsampled EOS at small-but-finite weight.
    """
    idx = np.asarray(eos_indices)
    idx = idx[np.isfinite(idx)].astype(int)
    idx = idx[(idx >= 0) & (idx < n_eos)]
    counts = np.bincount(idx, minlength=n_eos).astype(float) + floor_counts
    return np.log(counts / counts.sum())


def read_eos_indices_xml(samples_xml):
    """EOS indices from a CIP posterior-samples ligolw XML (sim_inspiral.alpha)."""
    from RIFT import lalsimutils                       # lazy: needs lal env
    plist = lalsimutils.xml_to_ChooseWaveformParams_array(str(samples_xml))
    out = []
    for P in plist:
        v = getattr(P, "eos_table_index", None)
        out.append(np.nan if v is None else float(v))
    return np.array(out)


def remap_net_index_column(net_in, net_out, n_eos, index_col=11):
    """Rewrite the composite's EOS-index column as ``index % n_eos`` (demo mode)."""
    dat = np.loadtxt(net_in)
    dat[:, index_col] = np.mod(dat[:, index_col].astype(int), n_eos)
    np.savetxt(net_out, dat)
    return net_out


class CIPTabularEvidence:
    """``evidence_fn(prior, coeffs) -> lnZ`` running one CIP job per round.

    Parameters
    ----------
    cip_exe   : path to ``util_ConstructIntrinsicPosterior_GenericCoordinates.py``
                (the patched RIFT with ``EOSSequenceFromFile``).
    net_file  : the trained likelihood composite (``all.net``-style).
    workdir   : scratch directory (one subdir per generation).
    rift_code : RIFT ``Code`` dir to prepend to PYTHONPATH for the subprocess.
    n_max, n_eff, n_output_samples, timeout, extra_args : CIP run knobs.
    remap_index_mod : rewrite the net's EOS-index column per round (demo mode --
                see module docstring).
    """

    def __init__(self, cip_exe, net_file, workdir, rift_code=None,
                 python_exe=None, args_template=DEFAULT_CIP_ARGS, extra_args="",
                 n_max=150000, n_eff=10, n_output_samples=4000,
                 remap_index_mod=True, floor_counts=0.1, timeout=1800,
                 verbose=False):
        self.cip_exe = str(cip_exe)
        self.net_file = str(net_file)
        self.workdir = str(workdir)
        self.rift_code = rift_code
        self.python_exe = python_exe or sys.executable
        self.args_template = args_template
        self.extra_args = extra_args
        self.n_max = int(n_max)
        self.n_eff = int(n_eff)
        self.n_output_samples = int(n_output_samples)
        self.remap_index_mod = remap_index_mod
        self.floor_counts = floor_counts
        self.timeout = timeout
        self.verbose = verbose
        self.last_run = None                            # diagnostics

    def __call__(self, prior, coeffs):
        n_eos = len(coeffs)
        gen = prior.generation
        rdir = os.path.join(self.workdir, f"gen_{gen}")
        os.makedirs(rdir, exist_ok=True)
        seq = os.path.join(rdir, "draws.pca.h5")
        write_pca_draws(seq, prior, coeffs)

        net = self.net_file
        if self.remap_index_mod:
            net = remap_net_index_column(
                self.net_file, os.path.join(rdir, "net_remapped.dat"), n_eos)

        samples_base = os.path.join(rdir, "posterior_samples")
        cmd = ([self.python_exe, self.cip_exe]
               + self.args_template.format(net=net, seq=seq).split()
               + ["--n-max", str(self.n_max), "--n-eff", str(self.n_eff),
                  "--fname-output-samples", samples_base,
                  "--n-output-samples", str(self.n_output_samples)]
               + (self.extra_args.split() if self.extra_args else []))
        env = dict(os.environ)
        if self.rift_code:
            env["PYTHONPATH"] = (str(self.rift_code) + ":"
                                 + env.get("PYTHONPATH", ""))
        res = subprocess.run(cmd, cwd=rdir, env=env, capture_output=True,
                             text=True, timeout=self.timeout)
        self.last_run = dict(returncode=res.returncode, dir=rdir,
                             tail=res.stdout[-2000:] + res.stderr[-2000:])
        if res.returncode != 0:
            raise RuntimeError(f"CIP failed (gen {gen}); see last_run tail:\n"
                               + self.last_run["tail"][-800:])
        idx = read_eos_indices_xml(samples_base + ".xml.gz")
        lnZ = posterior_mass_lnZ(idx, n_eos, floor_counts=self.floor_counts)
        if self.verbose:
            occ = int(np.sum(np.exp(lnZ) > 1.5 * self.floor_counts
                             / (len(idx) + n_eos * self.floor_counts)))
            print(f"[gen {gen}] CIP ok: {len(idx)} samples over {n_eos} EOS "
                  f"({occ} occupied); lnZ spread "
                  f"{np.max(lnZ)-np.min(lnZ):.2f}")
        return lnZ
