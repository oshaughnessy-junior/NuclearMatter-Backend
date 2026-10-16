"""Owned study protocol, fresh real-package source admission and deadline."""
from pathlib import Path
import sys, json, hashlib, datetime
ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT.parent
LOCK = json.loads((ROOT / 'LOCK.json').read_text())
PINS = json.loads((ROOT / 'SOURCE_PIN.json').read_text())
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def verify():
    for name, expected in PINS.items():
        if sha(CAMPAIGN / name) != expected:
            raise RuntimeError('Physical source changed; stop study: ' + name)
def preflight():
    verify()
    if datetime.datetime.now(datetime.timezone.utc) >= datetime.datetime(2026, 10, 16, 8, 50, tzinfo=datetime.timezone.utc):
        raise TimeoutError('No new scientific calls after locked08:50UTC')
verify()
sys.path.insert(0, str(CAMPAIGN / 'production_weak_spectral_port/test_mirror/src'))
from nmbackend.reference import weak_reference as W
assert Path(W.__file__).resolve() == (CAMPAIGN / 'production_weak_spectral_port/test_mirror/src/nmbackend/reference/weak_reference.py').resolve()
BATH = W.FreeElectronBath(**LOCK['bath'])
FINE = W.SpectralControls(**LOCK['controls_fine'])
def radiation(spec):
    return W.Radiation.empty() if spec['mode'] == 'empty' else W.Radiation.fd(spec['T'], spec['mu_nue'], spec['mu_antinue'])
def metadata():
    return dict(lock_sha256=sha(ROOT/'LOCK.json'), source_pin_sha256=sha(ROOT/'SOURCE_PIN.json'), candidate_manifest_sha256=LOCK['candidate_manifest_sha256'], bath=LOCK['bath'], model='fixed physical free fullGuo electronCC', source=W.__file__)
