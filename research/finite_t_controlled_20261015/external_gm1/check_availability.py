"""Read-only runtime availability and internal forward-contract manifest.

Does not install or execute an external EOS; no external numerical agreement
is implied by this artifact. Writes only the owned external_gm1 directory.
"""
from pathlib import Path
import sys,json,math,hashlib,importlib.util,shutil
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
prefixes=['/opt/homebrew/lib','/usr/local/lib','/Users/rossma/miniconda3/envs/junior_tools/lib']
environment=dict(executable=sys.executable,python_o2sclpy=str(importlib.util.find_spec('o2sclpy')),
                 o2scl_config=shutil.which('o2scl-config'),
                 checked_library_prefixes=prefixes,
                 library_candidates=[str(p) for root in prefixes for p in Path(root).glob('*o2scl*')])
path=ROOT/'rmf_reference/validation_results.json'
internal=json.loads(path.read_text())
contract=dict(status='EXTERNAL_FORWARD_GATE_OPEN',external_results=None,
    checked_utc=datetime.now(timezone.utc).isoformat(),
    target=dict(nmb_commit='25e68182f726e5bd9c0b372c9a667536940fbd07',
      M_MeV=939.,HBARC_MeV_fm=197.3269804,C_sigma_fm2=11.785,C_omega_fm2=7.148,
      C_rho_fm2=4.410,b=.002947,c=-.001070,
      composition='independent net nn,np; nuclear-only, uniform Hartree local branch',
      vacuum='no Dirac-sea term; thermal nucleon antiparticles retained'),
    o2scl_candidate_config=dict(cs_fm=math.sqrt(11.785),cw_fm=math.sqrt(7.148),cr_fm=math.sqrt(4.410),
      b=.002947,c=-.001070,neutron_m_fm_inv=939./197.3269804,proton_m_fm_inv=939./197.3269804,
      mnuc_fm_inv=939./197.3269804,degeneracy_each=2,inc_rest_mass=True,zm_mode=False,
      zero_fields=['zeta','xi','a1','a2','a3','a4','a5','a6','b1','b2','b3'],
      entrypoint='eos_had_rmf::calc_temp_e; T_in=T_MeV/HBARC; neutron.n=(1-Yp)nB; proton.n=Yp*nB'),
    environment=environment,
    internal_forward_source=str(path.relative_to(ROOT)),
    internal_forward_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    internal_reference_points_NOT_external=internal['sample_states'],
    acceptance_policy=['Pin external revision/config before execution',
      'Compare forward fields, masses, chemical potentials,e,P,s,f before derivatives',
      'Establish same local branch and independent density residuals',
      'Refine both quadratures/solver tolerances before assigning comparison error',
      'No saturation refit or parameter calibration to remove a discrepancy'])
(HERE/'comparison_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
print(json.dumps(dict(status=contract['status'],environment=environment),indent=2))
