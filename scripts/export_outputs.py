"""Export the two trusted local historical AirIO prediction files to portable NPZ."""
import argparse
import hashlib
import io
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import torch


class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=True)
        return super().find_class(module, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True, type=Path, help='trusted historical AirIO project root')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    entries = []
    for run in ['motion_body_rot', 'motion_body_freq_medium_rot']:
        source = args.source_root / 'experiments/euroc' / run / 'net_output.pickle'
        target = root / 'artifacts/predictions' / run
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target / 'net_output.pickle')
        # Only load files from this project's trusted historical runs.
        with source.open('rb') as f:
            original = CPUUnpickler(f).load()
        arrays = {}
        sequences = {}
        for sequence, fields in original.items():
            assert set(fields) == {'cov', 'net_vel', 'ts'}
            sequences[sequence] = {}
            for field, tensor in fields.items():
                array = tensor.detach().cpu().numpy()
                arrays[sequence + '__' + field] = array
                sequences[sequence][field] = {'shape': list(array.shape), 'dtype': str(array.dtype)}
        np.savez_compressed(target / 'predictions.npz', **arrays)
        entries.append({'run': run, 'source_relative': str(source.relative_to(args.source_root)),
                        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                        'sequences': sequences, 'conversion': 'tensor.detach().cpu().numpy(); no resampling or dtype cast',
                        'attitude_source': 'ground_truth', 'contains_raw_imu_or_ground_truth_states': False})
    (root / 'artifacts/predictions/manifest.json').write_text(json.dumps(entries, indent=2) + '\n')
    print('Exported 2 original pickle files and 2 portable NPZ files.')


if __name__ == '__main__':
    main()
