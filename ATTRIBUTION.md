# Attribution and provenance

`NuclearMatter-Backend` is a new (MIT-licensed) implementation. It was *motivated
by* and in places *based on* several prior works. This file records provenance so
credit is explicit and so we can confirm there are no licensing problems. Per
group convention, any module that reuses or re-implements an idea carries a
`# Based on:` or `# Motivated by:` note pointing back here.

## Summary of license status

| Source | Role here | License | Reuse status |
|---|---|---|---|
| **jester** (`jesterTOV`) | Differentiable JAX/diffrax TOV + EOS; conceptual template for the enthalpy formulation and the nuclear metamodel | **MIT** (© 2024 Peter Pang) | Safe to reuse/adapt with attribution. No code copied; formulation independently re-implemented. Data: `data/crust_sly_dh2001.dat` is the SLy crust table (Haensel, Ioffe; Douchin & Haensel 2001) as redistributed in `jesterTOV/crust_files/DH_crust.dat`, rows below 0.16 fm^-3. |
| **RePrimAnd** | High-accuracy reference TOV solver (validation target) | CC-BY / open (Zenodo 10617510) | Used only as an external optional backend / reference. Not vendored. |
| **RIFT** `EOSManager.py` | The `EOSConcrete` interface we adapt to | RIFT/RIT internal (no per-file header) | We target its public interface; no code copied. |
| **Zhu** `delta_tt` (`zzhu-astro/delta_tt`) and **fisher_eos** (LIGO GitLab `zhenyu.zhu/fisher_eos`) | Forward-mode (perturbation / "delta") TOV for tidal sensitivities; RMF nuclear parameterization (Esym, K0, Lsym, M*, J0); Fisher-matrix EOS application | **No LICENSE file present** | **Motivated by only.** No code copied. A compiled-C++ delta-TOV backend (Zhu) is planned behind `nmbackend.tov` once Z. Zhu is formally linked into this repository as a developer and a license is set. |
| **gwkokab** | Population-inference target (numpyro Distributions) | **Apache-2.0** | Compatible downstream; we provide adapters, not forks. |
| **diffrax** (Kidger) | Adaptive ODE backend for the default TOV solver | **Apache-2.0** | Core dependency; used as a library (not vendored). |
| **JAX** | Autodiff + array backend | **Apache-2.0** | Core dependency. |
| **MUSES** | Modern physical-EOS modules (RMFT, CompOSE) explored for the EOS layer | see project | Reference; integration planned. |
| **CompOSE** eos/116 and eos/188 (V. Dexheimer; Zenodo 10.5281/zenodo.14808482 and 10.5281/zenodo.14808749) | Real-file fixtures for the CompOSE reader and TOV tests (`tests/data/compose_eos116/`, `tests/data/compose_eos188/`) | **CC-BY-4.0** | Unmodified `eos.nb` and `eos.thermo` copies, redistributed with attribution and sha256 in each directory's README. |
| **Huth et al.** (Nature 606 (2022) 276, arXiv:2107.06229) & **Tsang et al.** (arXiv:2310.11588) | Density-tagged pressure-likelihood encoding of laboratory constraints (`lab_likelihood.py`) | published methods (papers; Huth samples on Zenodo not used) | **Motivated by only.** Encodings re-implemented from the papers; no code or data copied. |

## Key scientific references

The physics implemented or planned here draws on (non-exhaustive):

- **TOV / tidal deformability:** Tolman (1939); Oppenheimer & Volkoff (1939);
  Lindblom, *ApJ* 398 (1992) 569 (pseudo-enthalpy formulation);
  Hinderer, *ApJ* 677 (2008) 1216; Damour & Nagar, *PRD* 80 (2009) 084035;
  Postnikov, Prakash & Lattimer, *PRD* 82 (2010) 024016 (tidal Love number `y` ODE).
- **Nuclear empirical-parameter "metamodel":** Margueron, Hoffmann Casali &
  Gulminelli, *PRC* 97 (2018) 025805 & 025806.
- **Differentiable / AD EOS application:** Z. Zhu et al. (Fisher-matrix EOS
  inference; `fisher_eos`); see also arXiv:2604.05428, arXiv:2604.24949 (NN
  surrogates, rotating NS) cited in the project planning notes.
- **GP / nonparametric EOS:** Landry & Essick; Essick, Landry & Holz;
  Gorda et al. (e.g. arXiv:2605.18560).
- **JAX AD tooling:** `jester` (Wouters, Pang et al.).
- **Laboratory-likelihood encodings** (`lab_likelihood.py`): Huth et al.,
  *Nature* 606 (2022) 276 (arXiv:2107.06229) and Tsang et al.
  (arXiv:2310.11588) — density-tagged P(n) likelihoods; Biswas et al. /
  Margueron-line (S0, L) Gaussian priors as the parameter-space alternative.

## What is original here

- The JAX, `lax.scan` fixed-step RK4 TOV-in-enthalpy solver with in-loop tidal
  `y`-integration (`nmbackend/tov/solver.py`).
- The β-equilibrium nuclear metamodel with envelope-theorem pressure
  (`nmbackend/eos/metamodel.py`).
- The branch-aware M–R–Λ family with `dM/dh_c`, `dh_c/dM` (`nmbackend/tov/family.py`).
- The RIFT `EOSConcrete` adapter (`nmbackend/adapters/rift.py`).

If you reuse any of the *motivated-by* sources' actual code in the future, add the
corresponding license header and update the table above **before** committing.
