# Rotating-equilibrium development procedures

Select a host through infra-atlas before running science. The pilot uses
public equations and polytropes, with no observational data. No automation or
large production campaign is installed by these scripts.

## OSPool reference pilot

Build the external RNS driver on a compute/build host using `build_rns`.
For a portable x86-64 worker binary, append `-static` to the recorded compiler
command, rebuild, and record the new binary SHA and command separately. Do
not describe the dynamic binary's manifest as the static binary's provenance.
The November pilot used GNU/Linux 3.2 baseline, statically linked, without
architecture-specific instruction flags.

Stage that executable as `rns_export` alongside `osg_worker.sh` and
`osg_pilot.sub` in a new versioned directory on ap41. Check `quota -s` first.
The ten-task submit file is a bounded pilot, not a 40,000-star label-generation
campaign. RNS labels are unnecessary for the physics-only training loss.

```
mkdir results logs
condor_submit -dry-run pilot.ads osg_pilot.sub
condor_submit osg_pilot.sub
```

Each task requests one CPU, 512 MB memory and 64 MB disk, with a 120-second
solver timeout and 300-second running-job removal ceiling. Only the compressed
result archive transfers back on normal exit. The measured ten-task pilot
returned 5.8 MB and used 63 aggregate worker wall seconds. These are measured
pilot costs, not estimates for realistic EOS or finer grids. The ProjectName
is the atlas-documented existing OSPool project; recheck its applicability when
moving to another account or project.

After collection, verify hashes inside each archive and scheduler exit codes,
then run `qualify_corpus.py` on a compute host to validate exported fields.
Store failed/held outcomes alongside successful tasks. Do not retry an entire
cluster or modify unrelated held jobs.

## Scheduled development

SPORC's `nucleosynth` allocation was admitted on 6 November. The initial
CPU equation job used four cores, 8 GB memory and a ten-minute wall limit in
`interactive` with `qos_interactive`. Native x86 Python was
`/home/rossma/miniconda3/envs/genesis-gpu/bin/python` (JAX 0.10.1). Reverify
architecture and associations before using those facts. Tigris shares storage
but its aarch64 nodes cannot run this x86 environment.

A GPU pilot was admitted but delayed; it was cancelled after a small four-core
CPU pilot completed in 26 seconds on cardassia. Further bounded development
used that unrestricted host. Use large GPUs when the measured architecture
and batch need them, with an architecture-specific environment and explicit
Slurm resource/compute limits.

The commands and domain gates for physics checks, training, checkpoint loading
and RNS comparison are in `docs/rotating-pino-development.md`. Keep fresh run
paths and source snapshots. Reused RNS development cases are not an untouched
final test set. The next independent acceptance suite must use new cases and
predeclared forward and derivative tolerances.
