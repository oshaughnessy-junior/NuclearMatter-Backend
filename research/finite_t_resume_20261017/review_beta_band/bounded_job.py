from pathlib import Path
import sys,subprocess
P=Path(__file__).resolve().parent;i=int(sys.argv[1]);assert i in range(28)
try:r=subprocess.run([sys.executable,'-I','-B',str(P/'run_job.py'),str(i)],capture_output=True,text=True,timeout=90)
except subprocess.TimeoutExpired as e:(P/f'job{i}.TIMEOUT').write_text(repr(e));raise
(P/f'job{i}.stdout').write_text(r.stdout);(P/f'job{i}.stderr').write_text(r.stderr);print(r.stdout);assert r.returncode==0,r.stderr
