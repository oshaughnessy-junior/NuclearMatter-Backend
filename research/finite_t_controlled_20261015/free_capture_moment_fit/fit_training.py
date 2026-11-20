"""Freeze locked unweighted LS model using TRAINING data only."""
from pathlib import Path
import json,hashlib,datetime,numpy as np
P=Path(__file__).resolve().parent;lock=json.loads((P/'REVISED_LOCK2.json').read_text())
if(P/'COEFFICIENT_FREEZE.json').exists():raise RuntimeError('Frozen fit cannot be overwritten')
if list(P.glob('heldout_[0-9]*.json')):raise RuntimeError('Heldout data already exists: cannot assert training-only model lock')
rows=[json.loads((P/f'training_{i}.json').read_text())for i in range(11)]
if not all(r['qualified']is True for r in rows):raise RuntimeError('Unqualified training source')
X=np.array([r['basis']for r in rows]);Y=np.array([r['log_correction']for r in rows]);scale=np.linalg.norm(X,axis=0);cs,res,rank,sv=np.linalg.lstsq(X/scale,Y,rcond=None)
if rank!=7:raise ArithmeticError('Locked basis rank deficient; do not retune')
coef=cs/scale[:,None];errors=np.expm1(X@coef-Y)
out=dict(status='COEFFICIENTS_FROZEN_BEFORE_HELDOUT',locked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),lock_sha256=hashlib.sha256((P/'REVISED_LOCK2.json').read_bytes()).hexdigest(),coefficients=coef.tolist(),basis=lock['basis'],outputs=['R_EC','Q_EC','R_PC','Q_PC'],rank=int(rank),singular_values_scaled=sv.tolist(),column_norms=scale.tolist(),raw_condition=float(np.linalg.cond(X)),scaled_condition=float(np.linalg.cond(X/scale)),training_log_residuals=(X@coef-Y).tolist(),training_relative_errors=errors.tolist(),max_training_error=float(np.max(abs(errors))),training_files={f'training_{i}.json':hashlib.sha256((P/f'training_{i}.json').read_bytes()).hexdigest()for i in range(11)},heldout_files_before_freeze=[],method='Training-only unweighted linear LS; L2 column scaling then transformed back to originally locked basis.')
(P/'COEFFICIENT_FREEZE.json').write_text(json.dumps(out,indent=2)+'\n');print(out['max_training_error'],out['rank'])
