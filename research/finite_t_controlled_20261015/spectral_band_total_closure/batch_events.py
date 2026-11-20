from pathlib import Path
import json, subprocess, sys, datetime
R=Path(__file__).resolve().parent
for channel in ('EC','PC','NEUTRON_DECAY'):
    for scenario in range(3):
        if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime(2026,10,16,8,50,tzinfo=datetime.timezone.utc): raise TimeoutError('Locked stop; no new event call')
        command=[sys.executable,str(R/'event_worker.py'),channel,str(scenario)]
        try:
            result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=119)
        except subprocess.TimeoutExpired as error:
            (R/'events'/f'{channel}_{scenario}.TIMEOUT.txt').write_text(str(error));continue
        (R/'events'/f'{channel}_{scenario}.stdout.txt').write_text(result.stdout);print(result.stdout,flush=True)
        if result.returncode: raise RuntimeError('Event reporter/harness failed; preserve before repair')
