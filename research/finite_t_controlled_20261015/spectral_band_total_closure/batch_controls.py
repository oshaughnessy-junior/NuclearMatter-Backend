from pathlib import Path
import json, subprocess, sys
R = Path(__file__).resolve().parent
lock = json.loads((R/'LOCK.json').read_text())
for i, spec in enumerate(lock['small_inner_controls']):
    for j, var in enumerate(spec['variations']):
        cmd = [sys.executable, str(R/'control_worker.py'), str(i), str(j)]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=119)
        (R/'controls'/f'control_{i:02d}_{j}.stdout.txt').write_text(result.stdout)
        print(result.stdout, flush=True)
        if result.returncode: raise RuntimeError('Control harness failure; preserve before repair')
