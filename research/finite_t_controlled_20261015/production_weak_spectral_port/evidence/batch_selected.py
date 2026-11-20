from pathlib import Path
import subprocess,sys,json,time
R=Path(__file__).resolve().parents[1];fixture=json.loads((R/'staged/tests/data/free_weak_spectral_reference.json').read_text());lock=json.loads((R/'LOCK.json').read_text())
points=[(r['channel'],r['E_MeV'],'saved')for r in fixture['spectra']]+[(ch,E,'new')for ch,es in lock['new_points'].items()for E in es]
for ch,E,kind in points:
 for profile in ['base','fine']:
  name=f'{kind}_{ch}_{E}_{profile}';p=R/'evidence'/f'{name}.json'
  if p.exists():continue
  out=subprocess.run([sys.executable,str(R/'evidence/run_selected.py'),name,ch,str(E),profile],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=119)
  (R/'evidence'/f'{name}.stdout.txt').write_text(out.stdout)
  print(name,'exit',out.returncode,flush=True)
