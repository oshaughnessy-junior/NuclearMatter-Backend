"""Freeze the finite explicit import allowlist; rerunning is a NEW qualification."""
from pathlib import Path
import json,hashlib,sys,platform,importlib.metadata
P=Path(__file__).resolve().parent;R=P.parent
files=['hot_beta_moments/hot_moments.py','relativistic_crossed_contract/beta.py','relativistic_crossed_thermal/kernel.py','relativistic_free_eos/free_baryons.py','thermodynamics/lepton_reference.py','relativistic_tensor/tensor.py','relativistic_capture_oracle/events.py','muon_full_current/current.py','muon_vacuum/vacuum_cc.py','review_pc_moments/reference.py','isolated_microphysics/client.py','isolated_microphysics/protocol.py','isolated_microphysics/worker_entry.py']
pins=dict(sources={f:hashlib.sha256((R/f).read_bytes()).hexdigest()for f in files},runtime=dict(python=sys.version,numpy=importlib.metadata.version('numpy'),scipy=importlib.metadata.version('scipy'),platform=platform.platform()))
if __name__=='__main__':
 target=P/'PINS.json'
 if target.exists():raise RuntimeError('Refuse implicit repinning; preserve old identity and explicitly requalify')
 target.write_text(json.dumps(pins,indent=2))
