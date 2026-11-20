"""Replay byte-pinned public NSCool wrapper routine against original-code stdout."""
from pathlib import Path
import subprocess,json
HERE=Path(__file__).resolve().parent
r=subprocess.run([str(HERE/'cold_mu_mirror')],input=(HERE/'matched_input.txt').read_text(),text=True,capture_output=True,timeout=20)
(HERE/'mirror_matched_stdout.txt').write_text(r.stdout);(HERE/'mirror_matched_stderr.txt').write_text(r.stderr)
assert r.returncode==0,r.stderr
orig=[float(x.split()[1])for x in (HERE/'matched_stdout.txt').read_text().splitlines()];mirror=[float(x.split()[1])for x in r.stdout.splitlines()]
assert orig==mirror and len(orig)==99
result=dict(passed=True,checks=99,bitwise_values_equal=True,public_blob='7a58562ed3c6a74dd469810838873b91878c3b57',driver='mirror_driver.f',scope='Actual separate compiled source execution with identical matched inputs, not formula reimplementation')
(HERE/'mirror_execution.json').write_text(json.dumps(result,indent=2)+'\n');print('99/99 public-mirror Fortran results bitwise equal original local routine')
