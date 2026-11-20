from pathlib import Path
import subprocess,sys,json,datetime
P=Path(__file__).resolve().parent;ch,first,stamp,used,reset,account=sys.argv[1:];assert ch in ['EC','PC','NEUTRON_DECAY']and int(first)in range(0,20,4)
assert account=='82f4cfd9-9e20-47cf-9997-761328d89f60'and int(reset)==1792652325 and float(used)<48
now=datetime.datetime.now(datetime.timezone.utc);t=datetime.datetime.fromisoformat(stamp);assert 0<=(now-t).total_seconds()<60 and now<datetime.datetime.fromisoformat('2026-10-16T10:27:28+00:00')
(P/'raw'/f'{ch}_{int(first):02d}_QUOTA.json').write_text(json.dumps(dict(tool_guard_utc=stamp,used_percent=float(used),account=account,window_minutes=10080,resetsAt=int(reset)),indent=2)+'\n')
try:r=subprocess.run([sys.executable,'-I','-B',str(P/'nodes20.py'),ch,first],capture_output=True,text=True,timeout=119)
except subprocess.TimeoutExpired as e:
 (P/'raw'/f'{ch}_{int(first):02d}_TIMEOUT.txt').write_text(repr(e));raise
(P/'raw'/f'{ch}_{int(first):02d}.stdout').write_text(r.stdout);(P/'raw'/f'{ch}_{int(first):02d}.stderr').write_text(r.stderr);print(r.stdout);assert r.returncode==0,r.stderr
