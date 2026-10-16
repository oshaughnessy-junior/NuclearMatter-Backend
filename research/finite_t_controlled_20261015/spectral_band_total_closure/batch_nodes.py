from pathlib import Path
import sys,json,subprocess,datetime,time
R=Path(__file__).resolve().parent
channel=sys.argv[1]
lock=json.loads((R/'LOCK.json').read_text())
if channel not in lock['panels_MeV']:raise ValueError('Named locked channel')
start=time.monotonic();failures=[]
for order in lock['incident_orders']:
    stop_order=False
    for panel in range(5):
        for first in range(0,order,4):
            if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime(2026,10,16,8,50,tzinfo=datetime.timezone.utc):
                raise TimeoutError('Locked scientific stop; no new job')
            path=R/'raw'/f'{channel}_{order}_p{panel}_{first:02d}.json'
            if path.exists():
                existing=json.loads(path.read_text())
                if existing['complete']:continue
                raise RuntimeError('Prior incomplete job cannot be silently resumed')
            cmd=[sys.executable,str(R/'node_worker.py'),channel,str(order),str(panel),str(first)]
            try:
                proc=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=119)
            except subprocess.TimeoutExpired as error:
                (R/'raw'/f'{channel}_{order}_p{panel}_{first:02d}.TIMEOUT.txt').write_text(str(error))
                failures.append(dict(order=order,panel=panel,first=first,error='timeout',complete=False));stop_order=True;break
            (R/'raw'/f'{channel}_{order}_p{panel}_{first:02d}.stdout.txt').write_text(proc.stdout);print(proc.stdout,flush=True)
            if proc.returncode:raise RuntimeError('Harness failure; preserve before any repair')
            job=json.loads(path.read_text())
            if not job['complete']:
                failures.append(dict(order=order,panel=panel,first=first,error=job.get('error'),complete=False));stop_order=True;break
        if stop_order:break
(R/f'{channel}_JOB_STATUS.json').write_text(json.dumps(dict(channel=channel,failed_paths=failures,seconds=time.monotonic()-start,scope='Any missing/incomplete node forbids complete band assembly; no skip or zero-fill'),indent=2)+'\n')
print(channel,'finished',time.monotonic()-start,'failedpaths',len(failures),flush=True)
