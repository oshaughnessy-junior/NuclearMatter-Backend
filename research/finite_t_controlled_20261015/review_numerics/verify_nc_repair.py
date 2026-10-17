"""Reproduce all repaired NC review gates without overwriting initial evidence."""
from pathlib import Path
import runpy

here=Path(__file__).resolve().parent
probe=here/'probe_neutral_current.py'
source=probe.read_text().replace('neutral_current_review.json','neutral_current_repair_verification.json')
exec(compile(source,str(probe),'exec'),{'__file__':str(probe),'__name__':'__main__'})
runpy.run_path(str(here/'check_nc_entropy_repair.py'),run_name='__main__')
