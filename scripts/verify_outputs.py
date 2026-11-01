"""Check portable historical predictions without a GPU or dataset download."""
import json
from pathlib import Path

import numpy as np


root = Path(__file__).resolve().parents[1]
reports = []
for item in json.loads((root / 'artifacts/predictions/manifest.json').read_text()):
    path = root / 'artifacts/predictions' / item['run'] / 'predictions.npz'
    with np.load(path, allow_pickle=False) as data:
        for sequence in item['sequences']:
            vel, cov, ts = [data[sequence + '__' + k] for k in ['net_vel', 'cov', 'ts']]
            assert all(np.isfinite(x).all() for x in [vel, cov, ts])
            assert ts.dtype == np.float64, 'absolute timestamps must retain FP64'
            assert vel.shape[-1] == cov.shape[-1] == 3
            assert len(vel) == cov.shape[-2] == ts.shape[-2]
            delta = np.diff(ts.reshape(-1))
            assert np.all(delta >= 0), 'historical timestamps must not go backwards'
            duplicates = int(np.count_nonzero(delta == 0))
            assert np.all(cov > 0)
            reports.append({'run': item['run'], 'sequence': sequence, 'samples': len(vel),
                            'mean_speed_mps': float(np.linalg.norm(vel, axis=-1).mean()),
                            'timestamp_dtype': str(ts.dtype), 'duplicate_timestamps': duplicates,
                            'checks': 'passed_with_historical_warning' if duplicates else 'passed',
                            'warning': 'deduplicate equal timestamps before interpolation' if duplicates else None})
print(json.dumps(reports, indent=2))
