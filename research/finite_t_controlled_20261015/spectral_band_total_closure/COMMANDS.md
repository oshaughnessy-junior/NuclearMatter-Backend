# Reproduction and execution commands

Working directory: /Users/rossma/Genesis_LANL.
Scientific interpreter: /Users/rossma/miniconda3/envs/junior_tools/bin/python.
Environment: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1.

The completed scientific driver invocations were, serially, `batch_controls.py`,
`batch_nodes.py EC`, `batch_nodes.py PC`, `batch_nodes.py NEUTRON_DECAY`, and
`batch_events.py` in this directory. Each driver invokes owned worker scripts
with explicit channel/order/panel/chunk or radiation arguments; the driver source
records exact child commands and timeout119 seconds. Stored stdout is adjacent
to every child result. Batch drivers are checkpoint-aware and refuse overwriting
an incomplete result. Do not rerun frozen output paths. Science preflight rejects
new calls after 2026-10-16 08:50 UTC. An independent reviewer should use an owned
output directory and explicitly commissioned bounded calls instead.

Saved-only postprocessing used `postprocess.py` and `analyze.py` with their
explicit channel arguments as shown in each script, followed by `freeze.py`.
These read actual node/event JSON, verify source hashes, and do not call physical
coefficients. Both failed freeze harness versions/errors remain preserved;
current freeze.py only repairs metadata-key/count handling. Final manifest
includes all raw/log/error/harness bytes and excludes only its own identity,
final checkpoint, and generated __pycache__. No source, dependency, Git, paper,
plot, candidate or other study writes were performed.
